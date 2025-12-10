"""Session storage - persistence for evaluation sessions."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class SessionStorage:
    """Handles persistence for evaluation sessions."""

    base_dir: Path

    def __post_init__(self):
        """Ensure base directory exists."""
        self.base_dir = Path(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self, session_id: str) -> Path:
        """Get directory for a specific session."""
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def save_session(self, session_id: str, data: Dict[str, Any]) -> Path:
        """
        Save session data to JSON file.

        Args:
            session_id: Session identifier
            data: Session data dictionary

        Returns:
            Path to saved file
        """
        session_dir = self.get_session_dir(session_id)
        data_file = session_dir / "data.json"

        try:
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, cls=DateTimeEncoder)
            logger.info(f"Session saved: {data_file}")
            return data_file
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            raise

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from JSON file.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary or None if not found
        """
        session_dir = self.base_dir / session_id
        data_file = session_dir / "data.json"

        if not data_file.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            raise

    def save_intermediate(
        self,
        session_id: str,
        phase: str,
        data: Dict[str, Any]
    ) -> Path:
        """
        Save intermediate results for a specific phase.

        Args:
            session_id: Session identifier
            phase: Phase name (e.g., "pointwise", "pairwise")
            data: Phase data dictionary

        Returns:
            Path to saved file
        """
        session_dir = self.get_session_dir(session_id)
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        phase_file = raw_dir / f"{phase}.json"

        try:
            with open(phase_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, cls=DateTimeEncoder)
            logger.debug(f"Saved intermediate: {phase_file}")
            return phase_file
        except Exception as e:
            logger.error(f"Failed to save intermediate {phase}: {e}")
            raise

    def save_responses(
        self,
        session_id: str,
        responses: Dict[str, Any]
    ) -> Path:
        """
        Save raw responses from providers.

        Args:
            session_id: Session identifier
            responses: Dictionary of provider responses

        Returns:
            Path to saved file
        """
        session_dir = self.get_session_dir(session_id)
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        responses_file = raw_dir / "responses.json"

        try:
            with open(responses_file, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=2, cls=DateTimeEncoder)
            logger.debug(f"Saved responses: {responses_file}")
            return responses_file
        except Exception as e:
            logger.error(f"Failed to save responses: {e}")
            raise

    def list_sessions(self) -> list:
        """
        List all saved sessions.

        Returns:
            List of session IDs
        """
        sessions = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "data.json").exists():
                sessions.append(item.name)
        return sorted(sessions, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its files.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        session_dir = self.base_dir / session_id

        if not session_dir.exists():
            return False

        try:
            import shutil
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            raise
