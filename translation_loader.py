"""Load generated sign translation tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sign_actions import ACTIONS


DEFAULT_TRANSLATION_TABLE_PATH = Path(__file__).resolve().parent / "translation_tables.json"


@dataclass(frozen=True)
class Dialect:
    """A supported Philippine translation dialect."""

    code: str
    name: str
    language: str


class TranslationLoader:
    """Read and query generated translation tables."""

    def __init__(self, table_path: Path = DEFAULT_TRANSLATION_TABLE_PATH) -> None:
        """Load translation data from a generated JSON file."""
        self.table_path = table_path
        if not table_path.exists():
            raise FileNotFoundError(
                f"Translation table not found: {table_path}. "
                "Run `node scripts/build_translation_tables.mjs` first."
            )

        with table_path.open("r", encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)

        valid_actions = set(ACTIONS)
        translations = payload.get("translations", {})
        self._translations: dict[str, dict[str, str]] = {
            label: values
            for label, values in translations.items()
            if label in valid_actions and isinstance(values, dict)
        }
        dialects = payload.get("dialects", {})
        self._dialects = {
            code: Dialect(
                code=code,
                name=str(details.get("name", code)),
                language=str(details.get("language", details.get("locale", code))),
            )
            for code, details in dialects.items()
        }

    @property
    def dialects(self) -> dict[str, Dialect]:
        """Return configured dialect metadata keyed by dialect code."""
        return dict(self._dialects)

    def translate(self, label: str, dialect_code: str) -> str:
        """Return the translation for a model label, or a readable fallback."""
        if label not in ACTIONS:
            return ""
        translated = self._translations.get(label, {}).get(dialect_code, "").strip()
        if translated:
            return translated
        return label.replace("_", " ")

    def dialect_name(self, dialect_code: str) -> str:
        """Return a display name for a dialect code."""
        dialect = self._dialects.get(dialect_code)
        return dialect.name if dialect else dialect_code

    def speech_language(self, dialect_code: str) -> str:
        """Return the language tag requested by the selected dialect."""
        dialect = self._dialects.get(dialect_code)
        return dialect.language if dialect else dialect_code
