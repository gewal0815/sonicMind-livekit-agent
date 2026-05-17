"""SonicMind voice agent - explicit-dispatch LiveKit worker."""

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AgentServer, JobContext, JobProcess, cli

try:
    from livekit.agents import function_tool
except Exception:  # pragma: no cover - older livekit-agents builds still boot.
    def function_tool(fn=None, **_kwargs):
        if fn is None:
            return lambda inner: inner
        return fn

from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("sonicmind-agent")
logging.basicConfig(level=logging.INFO)

AGENT_NAME = "sonicmind-agent"
server = AgentServer()

logger.info(
    "SonicMind agent starting - LIVEKIT_URL=%s API_KEY_SET=%s OPENAI_KEY_SET=%s SUPABASE_SET=%s",
    os.getenv("LIVEKIT_URL", "(not set)"),
    bool(os.getenv("LIVEKIT_API_KEY")),
    bool(os.getenv("OPENAI_API_KEY")),
    bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
)

INSTRUCTIONS_BASE = (
    "You are SonicMind AI, a helpful voice assistant. "
    "Keep responses concise and conversational. "
    "Prefer short spoken answers unless the user asks for detail. "
    "Use the provided workspace report context and report lookup tools when available. "
    "Never expose or claim access to reports outside the current workspace."
)

INSTRUCTIONS_VOICE_AGENT = (
    "You are in a private 1-on-1 voice session with {user_name}. "
    "Answer their questions, summarize the conversation so far, and suggest follow-up points."
)

INSTRUCTIONS_MEETING = (
    "You have been invited into a live team meeting as an AI assistant. "
    "There may be multiple speakers. Listen carefully and attribute ideas to speakers when you can. "
    "You can summarize the discussion so far, identify action items mentioned, list open questions, "
    "or answer factual questions. Be concise. "
    "When the host says 'start daily' or 'start the daily', begin the daily standup flow."
)

_DAILY_START_TRIGGERS = {"start daily", "start the daily", "begin daily", "start standup", "begin standup"}


class SonicMindAssistant(Agent):
    def __init__(
        self,
        user_name: str = "there",
        room_type: str = "voice_agent",
        workspace_id: str | None = None,
        selected_report_id: str | None = None,
        report_context: dict | None = None,
    ) -> None:
        if room_type == "meeting":
            base_context = INSTRUCTIONS_MEETING
        else:
            base_context = INSTRUCTIONS_VOICE_AGENT.format(user_name=user_name)

        self.workspace_id = workspace_id
        self.selected_report_id = selected_report_id
        self.report_context = report_context if isinstance(report_context, dict) else {}
        self.reports = [
            report for report in self.report_context.get("reports", [])
            if isinstance(report, dict) and report.get("id")
        ]
        self.allowed_report_ids = set(self.report_context.get("allReportIds") or [r["id"] for r in self.reports])
        self.report_by_id = {str(report["id"]): report for report in self.reports}

        super().__init__(
            instructions=f"{INSTRUCTIONS_BASE} {base_context} {self._workspace_context_instructions()}",
        )

    def _workspace_context_instructions(self) -> str:
        if not self.workspace_id:
            return (
                "No workspace report context was provided for this call. "
                "You may still hold the live conversation and summarize what is said."
            )
        if not self.reports:
            return (
                f"This call is scoped to workspace {self.workspace_id}, but no reports were available. "
                "Do not invent report content."
            )

        primary = self.selected_report_id or self.report_context.get("primaryReportId")
        primary_text = (
            f"The primary selected report is {primary}. Use it first when the user says 'this report'. "
            if primary else
            "No primary report is selected. This is workspace/session mode; use any relevant workspace report when asked. "
        )
        catalog_lines = []
        for idx, report in enumerate(self.reports[:20], start=1):
            title = report.get("title") or "Untitled report"
            report_type = report.get("type") or "report"
            summary = report.get("summary") or report.get("preview") or ""
            marker = " PRIMARY" if primary and report.get("id") == primary else ""
            catalog_lines.append(f"{idx}. [{report.get('id')}] {title} ({report_type}){marker}: {summary}")

        return (
            f"The live call is scoped to workspace {self.workspace_id}. {primary_text}"
            "The available workspace report catalog follows. Use search_workspace_reports and "
            "get_workspace_report for precise details before answering report-specific questions. "
            + " ".join(catalog_lines)
        )

    @function_tool
    async def search_workspace_reports(self, query: str) -> str:
        """Search the current workspace report catalog by title, type, summary, and preview text."""
        q = (query or "").lower().strip()
        if not q:
            return "Please provide a search query."
        matches = []
        for report in self.reports:
            haystack = " ".join([
                str(report.get("title") or ""),
                str(report.get("type") or ""),
                str(report.get("summary") or ""),
                str(report.get("preview") or ""),
            ]).lower()
            if q in haystack:
                matches.append(report)
        if not matches:
            return "No matching reports were found in this workspace."
        return json.dumps([
            {
                "id": report.get("id"),
                "title": report.get("title"),
                "type": report.get("type"),
                "summary": report.get("summary") or report.get("preview"),
            }
            for report in matches[:10]
        ])

    @function_tool
    async def get_workspace_report(self, report_id: str) -> str:
        """Load a report by ID, restricted to reports injected for the current workspace."""
        report_id = (report_id or "").strip()
        if report_id not in self.allowed_report_ids:
            return "Access denied: that report is not part of the current workspace context."

        fetched = await _fetch_report_from_supabase(report_id, self.workspace_id)
        if fetched:
            return fetched

        cached = self.report_by_id.get(report_id)
        if not cached:
            return "Report not found in the current workspace context."
        return json.dumps(cached)


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
    workspace_id = metadata.get("workspaceId")
    selected_report_id = metadata.get("selectedReportId")
    report_context = metadata.get("reportContext") if isinstance(metadata.get("reportContext"), dict) else {}
    mode = resolve_mode(room_name)
    logger.info(
        "Starting %s session for room=%s room_type=%s user_id=%s workspace_id=%s reports=%d primary=%s",
        mode,
        room_name,
        room_type,
        metadata.get("userId"),
        workspace_id,
        len(report_context.get("allReportIds") or []),
        selected_report_id,
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
        agent=SonicMindAssistant(
            user_name=user_name,
            room_type=room_type,
            workspace_id=str(workspace_id) if workspace_id else None,
            selected_report_id=str(selected_report_id) if selected_report_id else None,
            report_context=report_context,
        ),
        room=ctx.room,
    )

    loop = asyncio.get_event_loop()

    @session.on("user_speech_committed")
    def on_any_user_speech(ev) -> None:
        transcript = _event_transcript(ev)
        if transcript:
            loop.create_task(_append_call_transcript(room_name, "user", transcript, metadata))

    @session.on("agent_speech_committed")
    def on_agent_speech(ev) -> None:
        transcript = _event_transcript(ev)
        if transcript:
            loop.create_task(_append_call_transcript(room_name, "agent", transcript, metadata))

    @ctx.room.on("data_received")
    def on_data_received(dp) -> None:
        try:
            raw = getattr(dp, "data", None)
            if raw is None:
                return
            msg = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
            msg_type = msg.get("type")

            if msg_type == "sonicmind_context":
                text = str(msg.get("text") or "")
                if text:
                    loop.create_task(session.generate_reply(instructions=text))
                    logger.info("Received SonicMind context packet (%d chars)", len(text))

            if room_type == "meeting" and msg_type == "DAILY_QUESTIONS":
                questions = msg.get("questions", [])
                title = str(msg.get("reportTitle") or "the report")
                if not questions:
                    return
                numbered = " ".join(f"Question {i + 1}: {q}" for i, q in enumerate(questions))
                instructions = (
                    f"The host has started the daily standup based on '{title}'. "
                    f"Read the following discussion questions aloud to the team, one by one. {numbered}"
                )
                loop.create_task(session.generate_reply(instructions=instructions))
                logger.info("Received DAILY_QUESTIONS (%d questions) for '%s'", len(questions), title)

            elif room_type == "meeting" and msg_type == "DAILY_COMPLETE":
                instructions = (
                    "Announce that the daily standup is complete and all answers have been saved to the report. "
                    "Keep it short."
                )
                loop.create_task(session.generate_reply(instructions=instructions))
                logger.info("Received DAILY_COMPLETE signal")
        except Exception as exc:
            logger.warning("data_received handler error: %s", exc)

    if room_type == "meeting":
        @session.on("user_speech_committed")
        def on_user_speech_daily(ev) -> None:
            try:
                transcript = _event_transcript(ev).lower().strip()
                if any(trigger in transcript for trigger in _DAILY_START_TRIGGERS):
                    logger.info("Detected 'Start Daily' voice command: %r", transcript)
                    loop.create_task(_publish_daily_start(ctx))
            except Exception as exc:
                logger.warning("user_speech_committed handler error: %s", exc)

        greeting = (
            "Greet the meeting participants. Say you are SonicMind AI and you are here as a meeting assistant. "
            "Mention that the host can say Start Daily or click the Start Daily button to begin the standup discussion."
        )
    else:
        if selected_report_id:
            greeting = (
                f"Greet {user_name if user_name != 'there' else 'the user'} and say you are connected to the selected report "
                "plus the rest of the workspace reports for context."
            )
        else:
            greeting = (
                f"Greet {user_name if user_name != 'there' else 'the user'} and say no single report is selected, "
                "so this conversation will be saved as an Audio Overview for the workspace session."
            )

    await session.generate_reply(instructions=greeting)


async def _publish_daily_start(ctx: JobContext) -> None:
    try:
        payload = json.dumps({"type": "DAILY_START"}).encode("utf-8")
        await ctx.room.local_participant.publish_data(payload, reliable=True)
        logger.info("Sent DAILY_START to room")
    except Exception as exc:
        logger.warning("Failed to publish DAILY_START: %s", exc)


def _event_transcript(ev) -> str:
    transcript = ""
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
    elif hasattr(ev, "text"):
        transcript = str(ev.text or "")
    return transcript.strip()


def _supabase_request(path: str, method: str = "GET", body: dict | None = None):
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not service_key:
        return None

    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{supabase_url}{path}",
        data=data,
        method=method,
        headers={
            "apikey": service_key,
            "authorization": f"Bearer {service_key}",
            "content-type": "application/json",
            "prefer": "return=representation",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else None


async def _fetch_report_from_supabase(report_id: str, workspace_id: str | None) -> str | None:
    if not workspace_id:
        return None

    def _fetch():
        encoded_id = urllib.parse.quote(report_id, safe="")
        rows = _supabase_request(
            f"/rest/v1/transcripts?id=eq.{encoded_id}&select=id,session_id,report_title,report_type,report_summary,text,timestamp&limit=1"
        )
        if not rows:
            return None
        row = rows[0]
        session_id = urllib.parse.quote(str(row.get("session_id") or ""), safe="")
        sessions = _supabase_request(f"/rest/v1/sessions?id=eq.{session_id}&select=workspace_id&limit=1")
        if not sessions or sessions[0].get("workspace_id") != workspace_id:
            return "Access denied: report is not in this workspace."
        return json.dumps(row)

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.warning("Failed to fetch report %s from Supabase: %s", report_id, exc)
        return None


async def _append_call_transcript(room_name: str, role: str, text: str, metadata: dict) -> None:
    if not text:
        return
    entry = {
        "role": role,
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "roomName": room_name,
        "userId": metadata.get("userId"),
        "workspaceId": metadata.get("workspaceId"),
        "sessionId": metadata.get("sessionId"),
    }

    def _append():
        room_name_filter = urllib.parse.quote(room_name, safe="")
        rows = _supabase_request(
            f"/rest/v1/livekit_call_recordings?room_name=eq.{room_name_filter}&select=transcript&limit=1"
        )
        current = []
        if rows and isinstance(rows[0].get("transcript"), list):
            current = rows[0]["transcript"]
        current.append(entry)
        _supabase_request(
            f"/rest/v1/livekit_call_recordings?room_name=eq.{room_name_filter}",
            method="PATCH",
            body={"transcript": current, "updated_at": datetime.now(timezone.utc).isoformat()},
        )

    try:
        await asyncio.to_thread(_append)
    except Exception as exc:
        logger.warning("Failed to append LiveKit transcript for %s: %s", room_name, exc)


if __name__ == "__main__":
    cli.run_app(server)
