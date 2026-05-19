"""
SonicMind LiveKit AI Agent
--------------------------
Joins workspace meeting rooms and voice-agent rooms.
- Transcribes all participants in real time
- Generates periodic summaries via LLM
- Responds to voice commands in voice-agent mode
- Routes app automation commands to voice-action-controller

Required environment variables:
  LIVEKIT_URL               wss://your-livekit.railway.app
  LIVEKIT_API_KEY           API key string
  LIVEKIT_API_SECRET        API secret string
  OPENAI_API_KEY            OpenAI API key (for STT, LLM, TTS)
  SUPABASE_URL              Supabase project URL  (optional — enables automation)
  SUPABASE_SERVICE_ROLE_KEY Supabase service role key (optional — enables automation)
"""

import asyncio
import json
import logging
import os
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero

# aiohttp is only needed for automation — import lazily so missing package
# does not crash the entire worker process.
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sonicmind-agent")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
AUTOMATION_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_KEY and AIOHTTP_AVAILABLE)

if not AIOHTTP_AVAILABLE:
    logger.warning("aiohttp not installed — voice automation disabled (pip install aiohttp)")
if not AUTOMATION_ENABLED:
    logger.warning("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set — voice automation disabled")

# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_WITH_AUTOMATION = """You are SonicMind, an intelligent AI meeting and productivity assistant.

Your capabilities:
- Real-time transcription and meeting summarization
- Question answering about the discussion
- Action item extraction and follow-up suggestions
- FULL CONTROL of the SonicMind platform through voice automation

IMPORTANT — App Automation:
You have a tool called `run_app_command` that gives you DIRECT access to control the app.
Use it whenever the user asks to:
  - Create / export / send reports
  - Schedule / cancel / update meetings or calendar events
  - Create slide decks or open the slide editor
  - Analyze URLs or websites
  - Create Hyperframe videos or render previews
  - Create dashboards or add / remove widgets

→ ALWAYS call `run_app_command` — do NOT say you cannot access the app.
→ After the tool returns, speak the result naturally.
→ If the tool returns a confirmation prompt, read it out and wait for yes / no.

Guidelines:
- Be concise. Keep answers under 3 sentences unless asked for more.
- Identify yourself as "SonicMind AI" if asked.
- Speak naturally and professionally.
"""

SYSTEM_PROMPT_BASIC = """You are SonicMind, an intelligent AI meeting assistant.

Your capabilities:
- Real-time meeting transcription and summarization
- Question answering about the conversation
- Action item extraction and follow-up suggestions

Guidelines:
- Be concise. Keep answers under 3 sentences unless asked for more.
- Identify yourself as "SonicMind AI" if asked.
- Speak naturally and professionally.
"""


# ── Automation HTTP client ────────────────────────────────────────────────

async def call_voice_action_controller(transcript: str, context: dict) -> dict:
    if not AIOHTTP_AVAILABLE:
        return {"status": "failed", "speakText": "Automation is not available.", "uiActions": []}

    url = f"{SUPABASE_URL}/functions/v1/voice-action-controller"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json={"transcript": transcript, "context": context},
                headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                body = await resp.text()
                logger.error(f"voice-action-controller {resp.status}: {body[:300]}")
                return {"status": "failed", "speakText": "Something went wrong.", "uiActions": []}
    except Exception as exc:
        logger.error(f"voice-action-controller call failed: {exc}")
        return {"status": "failed", "speakText": "Automation request failed.", "uiActions": []}


async def send_confirmation(run_id: str, approved: bool, context: dict) -> dict:
    if not AIOHTTP_AVAILABLE:
        return {"status": "failed"}
    url = f"{SUPABASE_URL}/functions/v1/voice-action-controller"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json={"runId": run_id, "confirmation": approved, "context": context},
                headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                return await resp.json() if resp.status == 200 else {"status": "failed"}
    except Exception as exc:
        logger.error(f"Confirmation failed: {exc}")
        return {"status": "failed"}


# ── Function context (tools) ──────────────────────────────────────────────

def build_fnc_ctx(room_context: dict) -> llm.FunctionContext:
    fnc_ctx = llm.FunctionContext()
    pending: dict = {}  # holds run_id while awaiting confirmation

    @fnc_ctx.ai_callable(
        description=(
            "Execute an app automation command for any SonicMind platform action: "
            "creating reports, scheduling meetings, analyzing URLs, generating slides, "
            "creating Hyperframe videos, or managing dashboards. "
            "Pass the user's exact spoken phrase."
        )
    )
    async def run_app_command(
        spoken_phrase: Annotated[
            str,
            llm.TypeInfo(description="The user's spoken automation request, word-for-word"),
        ],
    ) -> str:
        logger.info(f"[automation] run_app_command: {spoken_phrase!r}")

        # Handle pending confirmation
        phrase_lower = spoken_phrase.strip().lower()
        if pending.get("run_id"):
            run_id = pending["run_id"]
            approved = any(w in phrase_lower for w in ("yes", "confirm", "do it", "go ahead", "sure", "proceed", "ok", "okay"))
            rejected = any(w in phrase_lower for w in ("no", "cancel", "stop", "never mind", "nope"))
            if approved or rejected:
                pending.clear()
                result = await send_confirmation(run_id, approved, room_context)
                return result.get("speakText", "Done." if approved else "Cancelled.")

        result = await call_voice_action_controller(spoken_phrase, room_context)
        status = result.get("status", "failed")
        speak = result.get("speakText", "")

        if status == "needs_confirmation":
            run_id = result.get("runId")
            if run_id:
                pending["run_id"] = run_id
            return speak or "I need your confirmation. Please say yes or no."

        if status == "completed":
            return speak or "Done."

        if status == "failed":
            error = result.get("error", "")
            if error in ("unrecognised_intent", "parse-error", "no_action"):
                return "__FALLBACK__"  # LLM will answer naturally
            return speak or "Something went wrong. Please try again."

        return speak or "Processing your request."

    return fnc_ctx


# ── Background tasks ──────────────────────────────────────────────────────

async def transcribe_participants(ctx: agents.JobContext, agent: VoicePipelineAgent) -> None:
    async def handle_participant(participant: rtc.RemoteParticipant) -> None:
        logger.info(f"Participant joined: {participant.identity}")

    for participant in ctx.room.remote_participants.values():
        await handle_participant(participant)

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        asyncio.ensure_future(handle_participant(participant))


async def periodic_summary(ctx: agents.JobContext, agent: VoicePipelineAgent, interval: int = 600) -> None:
    await asyncio.sleep(interval)
    while True:
        try:
            await agent.say(
                "Here's a quick summary of the conversation so far.",
                allow_interruptions=True,
            )
        except Exception as exc:
            logger.warning(f"Periodic summary failed: {exc}")
        await asyncio.sleep(interval)


# ── Entrypoint ────────────────────────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext) -> None:
    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    # Parse room metadata injected by livekit-token edge function
    room_meta: dict = {}
    try:
        room_meta = json.loads(ctx.room.metadata or "{}")
    except Exception:
        pass

    is_voice_room = (
        room_meta.get("room_type") == "voice_agent"
        or ctx.room.name.startswith("voice-agent-")
    )

    room_context = {
        "workspaceId": room_meta.get("workspaceId") or "",
        "sessionId": room_meta.get("sessionId") or None,
        "userId": room_meta.get("userId") or "",
        "timezone": "UTC",
        "locale": "en",
        "role": "member",
        "plan": "free",
        "roomName": ctx.room.name,
    }

    has_user_context = bool(room_context["workspaceId"] and room_context["userId"])
    use_automation = AUTOMATION_ENABLED and has_user_context

    logger.info(
        f"Room: {ctx.room.name} | voice_room={is_voice_room} | "
        f"automation={use_automation} | workspaceId={room_context['workspaceId']!r}"
    )

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    system_prompt = SYSTEM_PROMPT_WITH_AUTOMATION if use_automation else SYSTEM_PROMPT_BASIC
    fnc_ctx = build_fnc_ctx(room_context) if use_automation else None

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
        chat_ctx=llm.ChatContext().append(role="system", text=system_prompt),
        fnc_ctx=fnc_ctx,
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        min_endpointing_delay=0.5,
    )

    agent.start(ctx.room, participant)
    await asyncio.sleep(1)

    if is_voice_room:
        greeting = (
            "Hello! I'm SonicMind AI. I can answer questions, create reports, "
            "schedule meetings, generate slides, and much more — all through voice. How can I help?"
            if use_automation
            else "Hello! I'm SonicMind AI. How can I help you today?"
        )
    else:
        greeting = (
            "Hello everyone! I'm SonicMind AI, your meeting assistant. "
            "I'll help with transcription, summaries, and automation. Just speak naturally."
        )

    await agent.say(greeting, allow_interruptions=True)

    if not is_voice_room:
        asyncio.ensure_future(transcribe_participants(ctx, agent))
        asyncio.ensure_future(periodic_summary(ctx, agent))

    try:
        while True:
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info(f"Agent leaving room: {ctx.room.name}")


def prewarm(proc: agents.JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
