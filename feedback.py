"""Persist post-session feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from session import SessionStats


DEFAULT_FEEDBACK_PATH = Path(__file__).resolve().parent / "feedback.json"


class FeedbackStore:
    """Append translator session feedback to a JSON file."""

    def __init__(self, feedback_path: Path = DEFAULT_FEEDBACK_PATH) -> None:
        """Create a feedback store for the provided JSON path."""
        self.feedback_path = feedback_path

    def append(
        self,
        session: SessionStats,
        satisfaction: int,
        comment: str,
    ) -> None:
        """Append one completed session without overwriting prior sessions."""
        payload = self._load_existing()
        entry = session.to_summary_dict()
        entry["satisfaction"] = satisfaction
        entry["comment"] = comment
        payload.append(entry)

        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
            file.write("\n")

    def _load_existing(self) -> list[dict[str, Any]]:
        if not self.feedback_path.exists():
            return []
        try:
            with self.feedback_path.open("r", encoding="utf-8") as file:
                existing = json.load(file)
        except json.JSONDecodeError:
            return []
        if isinstance(existing, list):
            return existing
        if isinstance(existing, dict):
            return [existing]
        return []
