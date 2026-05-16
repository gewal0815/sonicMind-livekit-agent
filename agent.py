"""SonicMind voice agent - explicit-dispatch LiveKit worker."""

import json
import logging

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AgentServer, JobContext, JobProcess, cli
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("sonicmind-agent")

AGENT_NAME = "sonicmind-agent"
server = AgentServer()


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
    "or answer factual questions. Be concise — you are a meeting assistant, not the focus."
)


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
        greeting = (
            "Greet the meeting participants briefly and say you're here as an AI assistant. "
            "Mention you can summarize the discussion, list action items, or answer questions."
        )
    else:
        greeting = (
            f"Greet {user_name if user_name != 'there' else 'the user'} and say you can answer questions, "
            "summarize this live conversation, and suggest follow-up points."
        )
    await session.generate_reply(instructions=greeting)


if __name__ == "__main__":
    cli.run_app(server)
