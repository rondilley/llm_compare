"""Session storage - persistence for evaluation sessions."""

import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Security: Valid session ID pattern (hex characters only)
SESSION_ID_PATTERN = re.compile(r'^[a-f0-9]{1,32}$')


class SecurityError(Exception):
    """Raised when a security validation fails."""
    pass


def _validate_session_id(session_id: str) -> None:
    """
    Validate that session ID contains only safe characters.

    Prevents path traversal attacks by ensuring session IDs
    are restricted to lowercase hex characters only.

    Args:
        session_id: Session identifier to validate

    Raises:
        SecurityError: If session_id contains invalid characters
    """
    if not session_id:
        raise SecurityError("Session ID cannot be empty")
    if not SESSION_ID_PATTERN.match(session_id):
        raise SecurityError(
            f"Invalid session ID format. Must contain only lowercase "
            f"hex characters (a-f, 0-9) and be 1-32 characters long."
        )


def _set_restrictive_permissions(path: Path) -> None:
    """
    Set restrictive file permissions (owner read/write only).

    On Unix: Sets mode to 0o600 (rw-------)
    On Windows: Attempts to restrict access via os.chmod

    Args:
        path: Path to file or directory
    """
    try:
        if sys.platform != 'win32':
            # Unix: Set strict permissions
            if path.is_dir():
                os.chmod(path, stat.S_IRWXU)  # 0o700 - rwx for owner only
            else:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600 - rw for owner only
        else:
            # Windows: Remove group/other access as much as possible
            # Note: Windows ACLs are more complex; this provides basic protection
            if path.is_dir():
                os.chmod(path, stat.S_IRWXU)
            else:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        # Log but don't fail - permissions may not be settable in all environments
        logger.warning(f"Could not set restrictive permissions on {path}: {e}")


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
        """
        Get directory for a specific session.

        Args:
            session_id: Session identifier (validated for security)

        Returns:
            Path to session directory

        Raises:
            SecurityError: If session_id contains invalid characters
        """
        _validate_session_id(session_id)
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        _set_restrictive_permissions(session_dir)
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
            _set_restrictive_permissions(data_file)
            logger.info(f"Session saved: {data_file}")
            return data_file
        except Exception as e:
            logger.error(f"Failed to save session: I/O error")
            raise

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from JSON file.

        Args:
            session_id: Session identifier (validated for security)

        Returns:
            Session data dictionary or None if not found

        Raises:
            SecurityError: If session_id contains invalid characters
        """
        _validate_session_id(session_id)
        session_dir = self.base_dir / session_id
        data_file = session_dir / "data.json"

        if not data_file.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session: I/O error")
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
            _set_restrictive_permissions(phase_file)
            logger.debug(f"Saved intermediate: {phase_file}")
            return phase_file
        except Exception as e:
            logger.error(f"Failed to save intermediate data: I/O error")
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
            _set_restrictive_permissions(responses_file)
            logger.debug(f"Saved responses: {responses_file}")
            return responses_file
        except Exception as e:
            logger.error(f"Failed to save responses: I/O error")
            raise

    def list_sessions(self) -> list:
        """List all saved session IDs."""
        return sorted(
            [item.name for item in self.base_dir.iterdir() if item.is_dir() and (item / "data.json").exists()],
            reverse=True
        )
