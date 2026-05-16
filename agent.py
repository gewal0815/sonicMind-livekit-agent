"""SonicMind voice agent - explicit-dispatch LiveKit worker."""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AgentServer, JobContext, JobProcess, cli
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("sonicmind-agent")
logging.basicConfig(level=logging.INFO)

AGENT_NAME = "sonicmind-agent"
server = AgentServer()

logger.info(
    "SonicMind agent starting — LIVEKIT_URL=%s API_KEY_SET=%s OPENAI_KEY_SET=%s",
    os.getenv("LIVEKIT_URL", "(not set)"),
    bool(os.getenv("LIVEKIT_API_KEY")),
    bool(os.getenv("OPENAI_API_KEY")),
)

INSTRUCTIONS_BASE = (
    "You are SonicMind AI, a helpful voice assistant. "
    "Keep responses concise and conversational. "
    "Prefer short spoken answers unless the user asks for detail. "
    "You cannot persist transcripts, read workspace files, create tasks, send emails, "
    "or produce reliable post-meeting artifacts yet. If asked for those, explain the limitation briefly."
)

INSTRUCTIONS_VOICE_AGENT = (
    "You are in a private 1-on-1 voice session with {user_name}. "
    "Answer their questions, summarize the conversation so far, and suggest follow-up points."
)

INSTRUCTIONS_MEETING = (
    "You have been invited into a live team meeting as an AI assistant. "
    "There may be multiple speakers. Listen carefully and attribute ideas to speakers when you can. "
    "You can: summarize the discussion so far, identify action items mentioned, list open questions, "
    "or answer factual questions. Be concise — you are a meeting assistant, not the focus. "
    "When the host says 'start daily' or 'start the daily', it means they want to begin the daily standup. "
    "You will then receive the discussion questions automatically and read them out loud to the team."
)

# Keywords that trigger sending DAILY_START back to the room
_DAILY_START_TRIGGERS = {"start daily", "start the daily", "begin daily", "start standup", "begin standup"}


class SonicMindAssistant(Agent):
    def __init__(self, user_name: str = "there", room_type: str = "voice_agent") -> None:
        if room_type == "meeting":
            context = INSTRUCTIONS_MEETING
        else:
            context = INSTRUCTIONS_VOICE_AGENT.format(user_name=user_name)

        super().__init__(
            instructions=f"{INSTRUCTIONS_BASE} {context}",
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def resolve_mode(room_name: str) -> str:
    if room_name.startswith("realtime-"):
        return "realtime"
    return "pipeline"


def get_job_metadata(ctx: JobContext) -> dict:
    raw_metadata = getattr(getattr(ctx, "job", None), "metadata", "") or "{}"
    try:
        parsed = json.loads(raw_metadata)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Invalid job metadata JSON: %s", raw_metadata)
        return {}


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    room_name = getattr(ctx.room, "name", "")
    metadata = get_job_metadata(ctx)
    room_type = str(metadata.get("roomType") or "voice_agent")
    user_name = str(metadata.get("userName") or "there")
    mode = resolve_mode(room_name)
    logger.info(
        "Starting %s session for room=%s room_type=%s user_id=%s",
        mode,
        room_name,
        room_type,
        metadata.get("userId"),
    )

    if mode == "realtime":
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )
    else:
        session = AgentSession(
            stt=openai.STT(model="whisper-1"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(model="tts-1", voice="alloy"),
            vad=ctx.proc.userdata["vad"],
        )

    await session.start(
        agent=SonicMindAssistant(user_name=user_name, room_type=room_type),
        room=ctx.room,
    )

    if room_type == "meeting":
        loop = asyncio.get_event_loop()

        # ── Data channel: receive messages from the meeting room ──────────────
        @ctx.room.on("data_received")
        def on_data_received(dp) -> None:
            try:
                raw = getattr(dp, "data", None)
                if raw is None:
                    return
                msg = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
                msg_type = msg.get("type")

                if msg_type == "DAILY_QUESTIONS":
                    questions: list = msg.get("questions", [])
                    title: str = msg.get("reportTitle", "the report")
                    if not questions:
                        return
                    numbered = " ".join(
                        f"Question {i + 1}: {q}" for i, q in enumerate(questions)
                    )
                    instructions = (
                        f"The host has started the daily standup based on '{title}'. "
                        f"Read the following discussion questions aloud to the team, one by one with a short pause between each. "
                        f"{numbered}"
                    )
                    loop.create_task(session.generate_reply(instructions=instructions))
                    logger.info("Received DAILY_QUESTIONS (%d questions) for '%s'", len(questions), title)

                elif msg_type == "DAILY_COMPLETE":
                    instructions = (
                        "Announce clearly that the daily standup is now complete and all answers have been saved to the report. "
                        "Congratulate the team and wish them a productive session. Keep it short — two sentences maximum."
                    )
                    loop.create_task(session.generate_reply(instructions=instructions))
                    logger.info("Received DAILY_COMPLETE signal")

            except Exception as exc:
                logger.warning("data_received handler error: %s", exc)

        # ── Voice command: detect 'Start Daily' in user speech ────────────────
        @session.on("user_speech_committed")
        def on_user_speech(ev) -> None:
            try:
                # The event may expose transcript via .transcript or via .message.content
                transcript: str = ""
                if hasattr(ev, "transcript"):
                    transcript = str(ev.transcript or "")
                elif hasattr(ev, "message") and hasattr(ev.message, "content"):
                    content = ev.message.content
                    if isinstance(content, list):
                        transcript = " ".join(
                            part.text if hasattr(part, "text") else str(part)
                            for part in content
                        )
                    else:
                        transcript = str(content or "")

                transcript_lower = transcript.lower().strip()
                if any(trigger in transcript_lower for trigger in _DAILY_START_TRIGGERS):
                    logger.info("Detected 'Start Daily' voice command: %r", transcript)
                    loop.create_task(_publish_daily_start(ctx))
            except Exception as exc:
                logger.warning("user_speech_committed handler error: %s", exc)

        greeting = (
            "Greet the meeting participants. Say you are SonicMind AI and you are here as a meeting assistant. "
            "Mention that the host can say 'Start Daily' or click the Start Daily button to begin the standup discussion, "
            "and that you can also summarize the conversation or answer questions at any time."
        )
    else:
        greeting = (
            f"Greet {user_name if user_name != 'there' else 'the user'} and say you can answer questions, "
            "summarize this live conversation, and suggest follow-up points."
        )

    await session.generate_reply(instructions=greeting)


async def _publish_daily_start(ctx: JobContext) -> None:
    """Send DAILY_START data packet to the room so the frontend opens the chat panel."""
    try:
        payload = json.dumps({"type": "DAILY_START"}).encode("utf-8")
        await ctx.room.local_participant.publish_data(payload, reliable=True)
        logger.info("Sent DAILY_START to room")
    except Exception as exc:
        logger.warning("Failed to publish DAILY_START: %s", exc)


if __name__ == "__main__":
    cli.run_app(server)
