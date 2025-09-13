import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from voicebridge.domain.models import SessionInfo
from voicebridge.ports.interfaces import SessionService


class FileSessionService(SessionService):
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self, audio_file: str, session_name: str | None = None
    ) -> SessionInfo:
        """Create a new transcription session."""
        session_id = str(uuid.uuid4())
        session = SessionInfo(
            session_id=session_id,
            session_name=session_name,
            created_at=datetime.now(),
            audio_file=audio_file,
            progress_seconds=0.0,
            total_duration=0.0,
            transcribed_text="",
            is_completed=False,
        )

        self.save_session(session)
        return session

    def save_session(self, session: SessionInfo) -> None:
        """Save session to disk."""
        session_file = self.sessions_dir / f"{session.session_id}.json"
        session_data = {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "created_at": session.created_at.isoformat(),
            "audio_file": session.audio_file,
            "progress_seconds": session.progress_seconds,
            "total_duration": session.total_duration,
            "transcribed_text": session.transcribed_text,
            "is_completed": session.is_completed,
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, session_id: str) -> SessionInfo:
        """Load session from disk."""
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            raise FileNotFoundError(f"Session {session_id} not found")

        with open(session_file) as f:
            session_data = json.load(f)

        return SessionInfo(
            session_id=session_data["session_id"],
            session_name=session_data.get("session_name"),
            created_at=datetime.fromisoformat(session_data["created_at"]),
            audio_file=session_data["audio_file"],
            progress_seconds=session_data.get("progress_seconds", 0.0),
            total_duration=session_data.get("total_duration", 0.0),
            transcribed_text=session_data.get("transcribed_text", ""),
            is_completed=session_data.get("is_completed", False),
        )

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions."""
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                session_id = session_file.stem
                session = self.load_session(session_id)
                sessions.append(session)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # Skip corrupted session files
                continue

        # Sort by creation time, newest first
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_file = self.sessions_dir / f"{session_id}.json"

        if session_file.exists():
            session_file.unlink()
            return True
        return False

    def cleanup_completed_sessions(self) -> int:
        """Remove completed sessions older than 30 days."""
        cleanup_count = 0
        cutoff_date = datetime.now() - timedelta(days=30)

        for session in self.list_sessions():
            if session.is_completed and session.created_at < cutoff_date:
                if self.delete_session(session.session_id):
                    cleanup_count += 1

        return cleanup_count

    def find_session_by_name(self, session_name: str) -> SessionInfo | None:
        """Find the most recent session with the given name."""
        sessions = self.list_sessions()

        for session in sessions:
            if session.session_name == session_name:
                return session

        return None

    def get_resumable_sessions(self) -> list[SessionInfo]:
        """Get sessions that can be resumed (not completed)."""
        sessions = self.list_sessions()
        return [s for s in sessions if not s.is_completed]
