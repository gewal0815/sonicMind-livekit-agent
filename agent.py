"""SonicMind voice agent — pipeline or realtime mode based on room name prefix."""

import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AgentServer, JobContext, JobProcess, cli
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("sonicmind-agent")

server = AgentServer()


class SonicMindAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are SonicMind, a helpful AI assistant in a workspace meeting. "
                "Keep responses concise and conversational. "
                "You can help summarize discussions, answer questions, and assist participants."
            ),
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def resolve_mode(room_name: str) -> str:
    if room_name.startswith("realtime-"):
        return "realtime"
    return "pipeline"


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    room_name = getattr(ctx.room, "name", "")
    mode = resolve_mode(room_name)
    logger.info("Starting %s session for room: %s", mode, room_name)

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

    await session.start(agent=SonicMindAssistant(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the participants and offer your assistance."
    )


if __name__ == "__main__":
    cli.run_app(server)
