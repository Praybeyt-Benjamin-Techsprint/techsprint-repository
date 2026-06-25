"""Train an LSTM model for sign language recognition.

The model consumes temporal landmark sequences shaped as:
    (samples, sequence_length, feature_count)

For full MediaPipe Holistic keypoints this is often (samples, 30, 1662).
For the hand-only collector in this project it is (samples, 30, 126).
The script infers the actual input shape from the loaded dataset so the same
training pipeline works for either feature set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

from preprocess_dataset import (
    ACTIONS,
    DATA_PATH,
    SEQUENCE_LENGTH,
    TEST_SIZE,
    discover_actions,
    preprocess_dataset,
)


LOG_DIR = os.path.join("Logs", "Train")
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.h5"
FINAL_MODEL_PATH = MODELS_DIR / "sign_language_model.h5"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
DEFAULT_EPOCHS = 100


def parse_args() -> argparse.Namespace:
    """Parse training configuration."""
    parser = argparse.ArgumentParser(
        description="Train an LSTM sign language recognition model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help=f"Dataset root directory. Default: {DATA_PATH}.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=f"Frames per sequence. Default: {SEQUENCE_LENGTH}.",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=ACTIONS,
        help="Action labels to train on. Defaults to the configured ACTIONS list.",
    )
    parser.add_argument(
        "--auto-actions",
        action="store_true",
        help="Use action folders found in DATA_PATH instead of the ACTIONS list.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Fraction of samples used for validation/testing. Default: {TEST_SIZE}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Maximum training epochs. Default: {DEFAULT_EPOCHS}.",
    )
    parser.add_argument(
        "--expected-features",
        type=int,
        default=None,
        help="Optional feature-width check, for example 1662 for full Holistic.",
    )
    return parser.parse_args()


def create_project_directories() -> None:
    """Create output directories for TensorBoard logs and saved models."""
    os.makedirs(os.path.join("Logs", "Train"), exist_ok=True)
    os.makedirs("models", exist_ok=True)


def create_callbacks() -> list:
    """Create TensorBoard, EarlyStopping, and ModelCheckpoint callbacks."""
    # TensorBoard tracks loss, accuracy, graph structure, and histograms so
    # training behavior can be inspected instead of guessed from final metrics.
    tb_callback = TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
    )

    # EarlyStopping helps prevent overfitting by stopping when validation loss
    # stops improving, then restoring the best observed model weights.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    checkpoint = ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_categorical_accuracy",
        save_best_only=True,
        verbose=1,
    )

    return [tb_callback, early_stopping, checkpoint]


def build_lstm_model(input_shape: tuple[int, int], num_classes: int) -> Sequential:
    """Build the LSTM architecture for sequence classification."""
    # Sign language is temporal: the class depends on how landmarks move across
    # frames, not just a single static pose. LSTMs are recurrent sequence models
    # that can learn motion patterns over time.
    model = Sequential(
        [
            Input(shape=input_shape),
            # Keras LSTM uses tanh by default. Keeping tanh is usually more
            # stable for recurrent state updates than forcing relu.
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            # Softmax outputs a probability distribution across gesture classes.
            Dense(num_classes, activation="softmax"),
        ]
    )

    # One-hot labels pair with categorical_crossentropy because the target for
    # each sample is a probability-like vector over all possible classes.
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def save_label_map(actions: list[str]) -> None:
    """Save the class-to-index mapping used during training."""
    label_map = {action: index for index, action in enumerate(actions)}
    LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2), encoding="utf-8")


def validate_feature_width(
    actual_features: int,
    expected_features: Optional[int],
) -> None:
    """Optionally enforce a specific per-frame feature width."""
    if expected_features is not None and actual_features != expected_features:
        raise ValueError(
            "Feature width mismatch: "
            f"dataset has {actual_features}, expected {expected_features}. "
            "Your hand-only collector produces 126 features per frame; full "
            "Holistic keypoints commonly produce 1662."
        )


def train_model(args: argparse.Namespace) -> None:
    """Load data, train the LSTM, and save model artifacts."""
    create_project_directories()

    actions = discover_actions(args.data_path) if args.auto_actions else list(args.actions)

    # Reuse the preprocessing pipeline to produce X_train, X_test, y_train, y_test.
    _, _, X_train, X_test, y_train, y_test = preprocess_dataset(args)
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    validate_feature_width(input_shape[1], args.expected_features)

    print(f"Training input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    model = build_lstm_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        callbacks=create_callbacks(),
    )

    model.save(str(FINAL_MODEL_PATH))
    save_label_map(actions)

    final_training_accuracy = history.history["categorical_accuracy"][-1]
    final_validation_accuracy = history.history["val_categorical_accuracy"][-1]

    print(f"Final training accuracy: {final_training_accuracy:.4f}")
    print(f"Final validation accuracy: {final_validation_accuracy:.4f}")
    print("Training Complete")
    print(f"Best Model Saved: {BEST_MODEL_PATH}")
    print(f"Final Model Saved: {FINAL_MODEL_PATH}")
    print("TensorBoard Command: tensorboard --logdir=Logs/Train")


def main() -> int:
    """Program entry point."""
    args = parse_args()
    try:
        train_model(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Training failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
