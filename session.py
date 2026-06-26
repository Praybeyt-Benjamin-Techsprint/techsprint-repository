"""Session statistics for accepted sign predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PredictionRecord:
    """One accepted prediction and its translated output."""

    original: str
    confidence: float
    translated: str
    timestamp: datetime

    def to_dict(self) -> dict[str, str | float]:
        """Serialize this record for feedback storage."""
        return {
            "original": self.original,
            "translated": self.translated,
            "confidence": round(self.confidence * 100, 2),
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
        }


@dataclass
class SessionStats:
    """Track statistics for one recognition session."""

    dialect: str
    started_at: datetime = field(default_factory=datetime.now)
    history: list[PredictionRecord] = field(default_factory=list)

    def add_prediction(
        self,
        label: str,
        confidence: float,
        translated_text: str,
        timestamp: datetime,
    ) -> None:
        """Record an accepted model prediction."""
        self.history.append(
            PredictionRecord(
                original=label,
                confidence=confidence,
                translated=translated_text,
                timestamp=timestamp,
            )
        )

    @property
    def prediction_count(self) -> int:
        """Return the number of accepted predictions."""
        return len(self.history)

    @property
    def average_confidence(self) -> float:
        """Return average confidence as a percentage."""
        if not self.history:
            return 0.0
        return sum(record.confidence for record in self.history) / len(self.history) * 100

    @property
    def highest_confidence(self) -> float:
        """Return highest confidence as a percentage."""
        if not self.history:
            return 0.0
        return max(record.confidence for record in self.history) * 100

    @property
    def lowest_confidence(self) -> float:
        """Return lowest confidence as a percentage."""
        if not self.history:
            return 0.0
        return min(record.confidence for record in self.history) * 100

    def to_summary_dict(self) -> dict[str, object]:
        """Serialize the session summary."""
        return {
            "date": self.started_at.isoformat(timespec="seconds"),
            "dialect": self.dialect,
            "predictions": self.prediction_count,
            "average_confidence": round(self.average_confidence, 2),
            "highest_confidence": round(self.highest_confidence, 2),
            "lowest_confidence": round(self.lowest_confidence, 2),
            "prediction_history": [record.to_dict() for record in self.history],
        }
