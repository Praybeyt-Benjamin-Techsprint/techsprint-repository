"""Frontend/controller for the Philippine Sign Language translator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sign_language
from feedback import FeedbackStore
from session import SessionStats
from sign_actions import ACTIONS, DELETE_ACTION, STOP_ACTION
from translation_loader import TranslationLoader
from tts import SystemTextToSpeechEngine, TextToSpeechEngine


STOP_COMMAND = STOP_ACTION
DELETE_COMMAND = DELETE_ACTION
CONTROL_COMMANDS = {
    STOP_COMMAND: "finalize_sentence",
    DELETE_COMMAND: "delete_last_word",
}

DIALECT_OPTIONS: list[tuple[str, str]] = [
    ("fil", "Filipino (Tagalog)"),
    ("ceb", "Cebuano"),
    ("ilo", "Ilokano"),
    ("pam", "Kapampangan"),
    ("hil", "Hiligaynon"),
]


@dataclass
class TranslatorApp:
    """Coordinate dialect selection, recognition, translation, and feedback."""

    translation_loader: TranslationLoader
    feedback_store: FeedbackStore
    tts_engine: TextToSpeechEngine

    def run(self) -> None:
        """Run the restartable translator menu."""
        dialect_code = self.choose_dialect()
        while True:
            self.run_session(dialect_code)
            action = self.restart_menu()
            if action == "1":
                continue
            if action == "2":
                dialect_code = self.choose_dialect()
                continue
            break

    def choose_dialect(self) -> str:
        """Prompt until a supported dialect is selected."""
        while True:
            print("==================================")
            print("Philippine Sign Language Translator")
            print("==================================")
            print()
            print("Choose a translation dialect:")
            print()
            for index, (_, name) in enumerate(DIALECT_OPTIONS, start=1):
                print(f"{index}. {name}")
            print()
            choice = input("Choice: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(DIALECT_OPTIONS):
                return DIALECT_OPTIONS[int(choice) - 1][0]
            print("Invalid choice. Please choose 1-5.")

    def run_session(self, dialect_code: str) -> None:
        """Start camera recognition and collect feedback after it exits."""
        dialect_name = self.translation_loader.dialect_name(dialect_code)
        session = SessionStats(dialect=dialect_name)
        english_buffer: list[str] = []
        translated_buffer: list[str] = []
        last_sign = ""
        last_confidence = 0.0

        self.print_live_display(
            dialect_name=dialect_name,
            english_sentence=english_buffer,
            translated_sentence=translated_buffer,
            last_sign=last_sign,
            confidence=last_confidence,
            model_status="Listening...",
        )

        def handle_prediction(prediction: sign_language.PredictionResult) -> bool:
            nonlocal last_sign, last_confidence

            label = prediction.label
            last_confidence = prediction.confidence

            command = CONTROL_COMMANDS.get(label)

            if command == CONTROL_COMMANDS[STOP_COMMAND]:
                completed_translated_sentence = self.build_sentence(translated_buffer)
                session.add_prediction(
                    label=label,
                    confidence=prediction.confidence,
                    translated_text=completed_translated_sentence,
                    timestamp=prediction.timestamp,
                )
                self.print_live_display(
                    dialect_name=dialect_name,
                    english_sentence=english_buffer,
                    translated_sentence=translated_buffer,
                    last_sign=last_sign,
                    confidence=last_confidence,
                    model_status="Stopped. Reading translated sentence...",
                )
                self.tts_engine.speak(
                    completed_translated_sentence,
                    self.translation_loader.speech_language(dialect_code),
                )
                english_buffer.clear()
                translated_buffer.clear()
                self.print_live_display(
                    dialect_name=dialect_name,
                    english_sentence=english_buffer,
                    translated_sentence=translated_buffer,
                    last_sign=last_sign,
                    confidence=last_confidence,
                    model_status="Sentence cleared. Listening...",
                )
                return True

            if command == CONTROL_COMMANDS[DELETE_COMMAND]:
                removed_translated_word = (
                    translated_buffer.pop() if translated_buffer else ""
                )
                if english_buffer:
                    english_buffer.pop()
                session.add_prediction(
                    label=label,
                    confidence=prediction.confidence,
                    translated_text="",
                    timestamp=prediction.timestamp,
                )
                self.print_live_display(
                    dialect_name=dialect_name,
                    english_sentence=english_buffer,
                    translated_sentence=translated_buffer,
                    last_sign=last_sign,
                    confidence=last_confidence,
                    model_status=(
                        "Last word removed."
                        if removed_translated_word
                        else "No words to remove."
                    ),
                )
                return True

            if label not in ACTIONS:
                self.print_live_display(
                    dialect_name=dialect_name,
                    english_sentence=english_buffer,
                    translated_sentence=translated_buffer,
                    last_sign=last_sign,
                    confidence=last_confidence,
                    model_status="Ignored unsupported prediction.",
                )
                return True

            last_sign = label
            english_word = self.format_english_word(prediction.label)
            translated = self.translation_loader.translate(prediction.label, dialect_code)
            english_buffer.append(english_word)
            translated_buffer.append(translated)
            session.add_prediction(
                label=prediction.label,
                confidence=prediction.confidence,
                translated_text=translated,
                timestamp=prediction.timestamp,
            )
            self.print_live_display(
                dialect_name=dialect_name,
                english_sentence=english_buffer,
                translated_sentence=translated_buffer,
                last_sign=last_sign,
                confidence=last_confidence,
                model_status="Listening...",
            )
            return True

        args = sign_language.parse_args()
        args.disable_threshold_slider = True
        sign_language.configure_logging(getattr(args, "debug", False))

        try:
            sign_language.run_webcam_loop(
                args,
                on_prediction=handle_prediction,
                translation_overlay_lines=lambda: self.translation_display_lines(
                    dialect_name=dialect_name,
                    english_sentence=english_buffer,
                    translated_sentence=translated_buffer,
                ),
                hidden_prediction_labels=set(CONTROL_COMMANDS),
            )
        except (sign_language.CameraOpenError, sign_language.CameraReadError) as exc:
            logging.error("Camera failure: %s", exc)
        except RuntimeError as exc:
            logging.error("Recognition failed: %s", exc)
        except KeyboardInterrupt:
            print()
            print("Session interrupted.")

        self.print_summary(session)
        satisfaction, comment = self.collect_feedback()
        self.feedback_store.append(session, satisfaction, comment)
        print("Feedback saved.")

    def print_live_display(
        self,
        dialect_name: str,
        english_sentence: list[str],
        translated_sentence: list[str],
        last_sign: str,
        confidence: float,
        model_status: str,
    ) -> None:
        """Refresh the terminal display for the current translation session."""
        english_text = self.build_sentence(english_sentence)
        translated_text = self.build_sentence(translated_sentence)
        print("\033[2J\033[H", end="")
        print("====================================")
        print(f"English Translation of FSL: {english_text or 'None'}")
        print(f"{dialect_name} Translation of FSL: {translated_text or 'None'}")
        print("====================================")
        print()
        print("Dialect")
        print(dialect_name)
        print()
        print("English")
        print(english_text or "None")
        print()
        print("Translated")
        print(translated_text or "None")
        print()
        print("------------------------------------")
        print()
        print("Last Sign")
        print(last_sign or "None")
        print()
        print("Confidence")
        print(f"{confidence * 100:.2f}%")
        print()
        print("Model")
        print(model_status)
        print()
        print("====================================")

    def translation_display_lines(
        self,
        dialect_name: str,
        english_sentence: list[str],
        translated_sentence: list[str],
    ) -> list[str]:
        """Return camera overlay lines for the current translation."""
        english_text = self.build_sentence(english_sentence)
        translated_text = self.build_sentence(translated_sentence)
        return [
            f"English Translation of FSL: {english_text or 'None'}",
            f"{dialect_name} Translation of FSL: {translated_text or 'None'}",
        ]

    @staticmethod
    def format_english_word(label: str) -> str:
        """Return a readable English display word for a model label."""
        return label.replace("_", " ").strip()

    @staticmethod
    def build_sentence(words: list[str]) -> str:
        """Join recognized words into a display sentence."""
        return " ".join(word for word in words if word).strip()

    def print_summary(self, session: SessionStats) -> None:
        """Display confidence statistics for the completed session."""
        print()
        print("Session Summary")
        print()
        print("Predictions:")
        print(session.prediction_count)
        print()
        print("Average Confidence:")
        print(f"{session.average_confidence:.1f}%")
        print()
        print("Highest Confidence:")
        print(f"{session.highest_confidence:.1f}%")
        print()
        print("Lowest Confidence:")
        print(f"{session.lowest_confidence:.1f}%")

    def collect_feedback(self) -> tuple[int, str]:
        """Ask for satisfaction score and optional comments."""
        print()
        print("How satisfied were you?")
        print()
        print("1 - Very Unsatisfied")
        print("2 - Unsatisfied")
        print("3 - Neutral")
        print("4 - Satisfied")
        print("5 - Very Satisfied")
        print()
        while True:
            choice = input("Choice: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= 5:
                break
            print("Invalid choice. Please choose 1-5.")
        print()
        comment = input("Additional comments: ").strip()
        return int(choice), comment

    def restart_menu(self) -> str:
        """Prompt for the next action after feedback is saved."""
        print()
        print("What would you like to do?")
        print()
        print("1. Start another session")
        print("2. Choose another dialect")
        print("3. Exit")
        print()
        while True:
            choice = input("Choice: ").strip()
            if choice in {"1", "2", "3"}:
                return choice
            print("Invalid choice. Please choose 1, 2, or 3.")


def main() -> int:
    """Program entry point."""
    app = TranslatorApp(
        translation_loader=TranslationLoader(),
        feedback_store=FeedbackStore(),
        tts_engine=SystemTextToSpeechEngine(),
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
