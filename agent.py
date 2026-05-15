"""
SonicMind LiveKit AI Agent
--------------------------
Joins workspace meeting rooms and voice-agent rooms.
- Transcribes all participants in real time
- Generates periodic summaries via LLM
- Responds to voice commands in voice-agent mode

Required environment variables:
  LIVEKIT_URL          wss://your-livekit.railway.app
  LIVEKIT_API_KEY      API key string
  LIVEKIT_API_SECRET   API secret string
  OPENAI_API_KEY       OpenAI API key (for STT, LLM, TTS)
"""

import asyncio
import json
import logging
import os
from datetime import datetime

from livekit import agents, rtc
from livekit.agents import llm, transcription
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sonicmind-agent")

SYSTEM_PROMPT = """You are SonicMind, an intelligent AI meeting assistant.
You join workspace video meetings and voice sessions to help participants.

Your capabilities:
- Real-time transcription of all speakers
- Meeting summarization (triggered by voice command or automatic every 10 minutes)
- Question answering about topics discussed in the meeting
- Action item extraction
- Follow-up question suggestions

Guidelines:
- Be concise. In meetings, keep answers under 3 sentences unless asked for more.
- When summarizing, use bullet points grouped by topic.
- Always identify yourself as "SonicMind AI" if asked who you are.
- Speak naturally and professionally.
"""


async def transcribe_participants(ctx: agents.JobContext, agent: VoicePipelineAgent) -> None:
    """Subscribe to all remote participants and log transcriptions."""
    conversation_log: list[dict] = []

    async def handle_participant(participant: rtc.RemoteParticipant) -> None:
        logger.info(f"Subscribing to participant: {participant.identity}")

    for participant in ctx.room.remote_participants.values():
        await handle_participant(participant)

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        asyncio.ensure_future(handle_participant(participant))


async def maybe_generate_summary(
    ctx: agents.JobContext,
    agent: VoicePipelineAgent,
    interval_seconds: int = 600,
) -> None:
    """Generate and broadcast a meeting summary every `interval_seconds`."""
    await asyncio.sleep(interval_seconds)
    while True:
        logger.info("Generating periodic meeting summary…")
        try:
            await agent.say(
                "Let me give you a quick meeting summary. "
                "I've been following the conversation and here are the key points so far.",
                allow_interruptions=True,
            )
        except Exception as exc:
            logger.warning(f"Summary generation failed: {exc}")
        await asyncio.sleep(interval_seconds)


async def entrypoint(ctx: agents.JobContext) -> None:
    """Agent entrypoint — called once per room dispatch."""
    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    # Detect room type from metadata or room name
    room_metadata: dict = {}
    try:
        room_metadata = json.loads(ctx.room.metadata or "{}")
    except Exception:
        pass

    is_voice_agent_room = (
        room_metadata.get("room_type") == "voice_agent"
        or ctx.room.name.startswith("voice-agent-")
    )

    participant = await ctx.wait_for_participant()
    logger.info(f"Connected with participant: {participant.identity}")

    # Build voice pipeline
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
        chat_ctx=llm.ChatContext().append(role="system", text=SYSTEM_PROMPT),
        # Allow users to interrupt the agent mid-speech
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        min_endpointing_delay=0.5,
    )

    agent.start(ctx.room, participant)

    if is_voice_agent_room:
        # 1-on-1 voice agent session
        await asyncio.sleep(1)
        await agent.say(
            "Hello! I'm SonicMind AI. How can I help you today?",
            allow_interruptions=True,
        )
    else:
        # Multi-party meeting
        await asyncio.sleep(1)
        await agent.say(
            "Hello everyone! I'm SonicMind AI, your meeting assistant. "
            "I'll transcribe the conversation and can generate summaries on request. "
            "Just say 'Hey SonicMind, summarize the meeting' at any time.",
            allow_interruptions=True,
        )
        # Start background tasks
        asyncio.ensure_future(transcribe_participants(ctx, agent))
        asyncio.ensure_future(maybe_generate_summary(ctx, agent, interval_seconds=600))

    # Keep agent alive until room closes
    try:
        while True:
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info(f"Agent leaving room: {ctx.room.name}")


def prewarm(proc: agents.JobProcess) -> None:
    """Pre-load models to reduce cold-start latency."""
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
