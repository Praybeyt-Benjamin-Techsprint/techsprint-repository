"""Text-to-speech abstraction for translated signs."""

from __future__ import annotations

import platform
import shutil
import subprocess
from abc import ABC, abstractmethod


class TextToSpeechEngine(ABC):
    """Interface for pluggable text-to-speech engines."""

    @abstractmethod
    def speak(self, text: str, language: str) -> None:
        """Speak text in the requested language when the engine supports it."""


class SystemTextToSpeechEngine(TextToSpeechEngine):
    """Use the operating system's available speech command when present."""

    def speak(self, text: str, language: str) -> None:
        """Speak text and wait until the speech command finishes."""
        if not text.strip():
            return

        command = self._build_command(text, language)
        if command is None:
            return

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.wait()
        except OSError:
            return

    def _build_command(self, text: str, language: str) -> list[str] | None:
        system = platform.system()
        if system == "Darwin" and shutil.which("say"):
            return ["say", text]
        if system == "Windows" and shutil.which("powershell"):
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$speaker.Speak({text!r})"
            )
            return ["powershell", "-NoProfile", "-Command", script]
        if shutil.which("spd-say"):
            return ["spd-say", "--wait", "-l", language, text]
        if shutil.which("espeak"):
            return ["espeak", "-v", language, text]
        return None


def speak(text: str, language: str) -> None:
    """Speak text using the default system TTS engine."""
    SystemTextToSpeechEngine().speak(text, language)
