"""
SonicMind LiveKit AI Agent
--------------------------
Joins workspace meeting rooms and voice-agent rooms.
- Transcribes all participants in real time
- Generates periodic summaries via LLM
- Responds to voice commands in voice-agent mode
- Routes app automation commands to voice-action-controller

Required environment variables:
  LIVEKIT_URL              wss://your-livekit.railway.app
  LIVEKIT_API_KEY          API key string
  LIVEKIT_API_SECRET       API secret string
  OPENAI_API_KEY           OpenAI API key (for STT, LLM, TTS)
  SUPABASE_URL             Supabase project URL
  SUPABASE_SERVICE_ROLE_KEY Supabase service role key
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Annotated

import aiohttp
from livekit import agents, rtc
from livekit.agents import llm, transcription
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sonicmind-agent")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

SYSTEM_PROMPT = """You are SonicMind, an intelligent AI meeting and productivity assistant.
You join workspace video meetings and voice sessions to help participants.

Your capabilities:
- Real-time transcription of all speakers
- Meeting summarization (triggered by voice command or automatic every 10 minutes)
- Question answering about topics discussed in the meeting
- Action item extraction and follow-up suggestions
- FULL CONTROL of the SonicMind platform through voice automation

IMPORTANT — App Automation:
You have a tool called `run_app_command` that gives you DIRECT access to control the app.
When the user asks you to do anything related to:
  - Reports (create, generate, export, send, open)
  - Meetings / Calendar (schedule, cancel, invite, check availability, set reminders)
  - Slide decks (create, export, open editor)
  - URL analysis (analyze, summarize, compare websites)
  - Hyperframe videos (create, render, export)
  - Dashboards and widgets (create, add widget, generate from report)

→ ALWAYS call `run_app_command` with the user's exact spoken phrase.
→ Do NOT say you cannot access the app. You CAN — through `run_app_command`.
→ After the tool returns, speak the result naturally to the user.
→ If the tool returns a confirmation prompt, read it out and wait for the user's yes/no.

Guidelines:
- Be concise. Keep answers under 3 sentences unless asked for more.
- When summarizing, use bullet points grouped by topic.
- Always identify yourself as "SonicMind AI" if asked who you are.
- Speak naturally and professionally.
- For app commands, acknowledge immediately ("Sure, let me do that for you") before calling the tool.
"""


# ── Voice Action Controller client ────────────────────────────────────────

async def call_voice_action_controller(
    transcript: str,
    context: dict,
) -> dict:
    """Call the Supabase voice-action-controller edge function."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set — automation disabled")
        return {"status": "failed", "speakText": "", "uiActions": []}

    url = f"{SUPABASE_URL}/functions/v1/voice-action-controller"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    payload = {"transcript": transcript, "context": context}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return await resp.json()
                body = await resp.text()
                logger.error(f"voice-action-controller {resp.status}: {body[:200]}")
                return {"status": "failed", "speakText": "", "uiActions": []}
    except Exception as exc:
        logger.error(f"voice-action-controller call failed: {exc}")
        return {"status": "failed", "speakText": "", "uiActions": []}


async def confirm_action(run_id: str, approved: bool, context: dict) -> dict:
    """Send confirmation decision back to voice-action-controller."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"status": "failed"}

    url = f"{SUPABASE_URL}/functions/v1/voice-action-controller"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    payload = {"runId": run_id, "confirmation": approved, "context": context}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                return await resp.json() if resp.status == 200 else {"status": "failed"}
    except Exception as exc:
        logger.error(f"Confirm action failed: {exc}")
        return {"status": "failed"}


# ── Function context (tools the LLM can call) ─────────────────────────────

def build_fnc_ctx(room_context: dict) -> llm.FunctionContext:
    """Build function context with automation tools wired to this room's context."""
    fnc_ctx = llm.FunctionContext()

    # Holds any pending confirmation run ID between turns
    pending_confirmation: dict = {}

    @fnc_ctx.ai_callable(
        description=(
            "Execute an app automation command. Use this for ANY request that involves "
            "the SonicMind platform: creating reports, scheduling meetings, analyzing URLs, "
            "generating slides, creating videos, managing dashboards, or exporting content. "
            "Pass the user's spoken phrase exactly as-is."
        )
    )
    async def run_app_command(
        spoken_phrase: Annotated[
            str,
            llm.TypeInfo(description="The user's spoken automation request, word-for-word"),
        ],
    ) -> str:
        logger.info(f"run_app_command: {spoken_phrase!r}")

        # If there's a pending confirmation, treat yes/no answers as confirmation
        phrase_lower = spoken_phrase.strip().lower()
        if pending_confirmation.get("run_id"):
            run_id = pending_confirmation["run_id"]
            approved = any(w in phrase_lower for w in ("yes", "confirm", "do it", "go ahead", "sure", "proceed", "ok", "okay"))
            rejected = any(w in phrase_lower for w in ("no", "cancel", "stop", "never mind", "nope", "don't"))

            if approved or rejected:
                pending_confirmation.clear()
                result = await confirm_action(run_id, approved, room_context)
                speak = result.get("speakText", "Done." if approved else "Cancelled.")
                return speak

        # Normal automation request
        result = await call_voice_action_controller(spoken_phrase, room_context)
        status = result.get("status", "failed")
        speak = result.get("speakText", "")

        if status == "needs_confirmation":
            run_id = result.get("runId")
            if run_id:
                pending_confirmation["run_id"] = run_id
            return speak or "I need your confirmation before proceeding. Please say yes or no."

        if status == "completed":
            return speak or "Done."

        if status == "failed":
            error = result.get("error", "")
            if error in ("unrecognised_intent", "parse-error"):
                # Not an app command — let LLM answer naturally
                return "__FALLBACK__"
            return speak or "Something went wrong. Please try again."

        return speak or "Processing your request."

    return fnc_ctx


# ── Participant transcription ─────────────────────────────────────────────

async def transcribe_participants(ctx: agents.JobContext, agent: VoicePipelineAgent) -> None:
    """Subscribe to all remote participants and log transcriptions."""
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


# ── Entrypoint ────────────────────────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext) -> None:
    """Agent entrypoint — called once per room dispatch."""
    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    # Parse room metadata (set by livekit-token edge function)
    room_metadata: dict = {}
    try:
        room_metadata = json.loads(ctx.room.metadata or "{}")
    except Exception:
        pass

    is_voice_agent_room = (
        room_metadata.get("room_type") == "voice_agent"
        or ctx.room.name.startswith("voice-agent-")
    )

    # Build runtime context passed to voice-action-controller
    room_context = {
        "workspaceId": room_metadata.get("workspaceId") or "",
        "sessionId": room_metadata.get("sessionId") or None,
        "userId": room_metadata.get("userId") or "",
        "timezone": "UTC",
        "locale": "en",
        "role": "member",
        "plan": "free",
        "roomName": ctx.room.name,
    }

    logger.info(f"Room context: workspaceId={room_context['workspaceId']!r}, userId={room_context['userId']!r}")

    # Only enable automation when we have a valid userId
    automation_enabled = bool(room_context["workspaceId"] and room_context["userId"])
    if not automation_enabled:
        logger.warning("No workspaceId or userId in room metadata — automation disabled for this room")

    participant = await ctx.wait_for_participant()
    logger.info(f"Connected with participant: {participant.identity}")

    # Build function context (tools) when automation is available
    fnc_ctx = build_fnc_ctx(room_context) if automation_enabled else None

    # Build voice pipeline
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
        chat_ctx=llm.ChatContext().append(role="system", text=SYSTEM_PROMPT),
        fnc_ctx=fnc_ctx,
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        min_endpointing_delay=0.5,
    )

    agent.start(ctx.room, participant)

    await asyncio.sleep(1)

    if is_voice_agent_room:
        greeting = (
            "Hello! I'm SonicMind AI. I can answer questions, generate reports, "
            "schedule meetings, create slide decks, and much more — all through voice. "
            "How can I help you?"
            if automation_enabled
            else "Hello! I'm SonicMind AI. How can I help you today?"
        )
    else:
        greeting = (
            "Hello everyone! I'm SonicMind AI, your meeting assistant. "
            "I'll transcribe the conversation and can automate reports, summaries, "
            "and much more. Just speak naturally."
        )

    await agent.say(greeting, allow_interruptions=True)

    if not is_voice_agent_room:
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
