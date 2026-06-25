"""Save, reload, and verify a trained sign language recognition model.

This script demonstrates model persistence for a TensorFlow/Keras LSTM model:

1. Load an existing trained model.
2. Save the full model to models/action.h5.
3. Delete the model object from memory.
4. Reload the full model with load_model().
5. Verify predictions against X_test/y_test.

Full-model loading is the preferred path for inference because it restores the
architecture and trained weights together, avoiding manual reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential, load_model

from preprocess_dataset import ACTIONS, DATA_PATH, SEQUENCE_LENGTH, TEST_SIZE
from preprocess_dataset import preprocess_dataset


DEFAULT_SOURCE_MODEL_PATH = Path("models/best_model.h5")
DEFAULT_ACTION_MODEL_PATH = Path("models/action.h5")
DEFAULT_LABEL_MAP_PATH = Path("models/label_map.json")
DEFAULT_SAMPLE_INDEX = 4


def parse_args() -> argparse.Namespace:
    """Parse inference verification options."""
    parser = argparse.ArgumentParser(
        description="Save, reload, and verify a trained sign language model."
    )
    parser.add_argument(
        "--source-model-path",
        type=Path,
        default=DEFAULT_SOURCE_MODEL_PATH,
        help=f"Existing trained model to copy from. Default: {DEFAULT_SOURCE_MODEL_PATH}.",
    )
    parser.add_argument(
        "--action-model-path",
        type=Path,
        default=DEFAULT_ACTION_MODEL_PATH,
        help=f"Full model output path. Default: {DEFAULT_ACTION_MODEL_PATH}.",
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=DEFAULT_LABEL_MAP_PATH,
        help=f"JSON label map created during training. Default: {DEFAULT_LABEL_MAP_PATH}.",
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
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Fraction of samples used for testing. Default: {TEST_SIZE}.",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=None,
        help="Fallback action labels if models/label_map.json is unavailable.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help=f"Test sample index to verify. Default: {DEFAULT_SAMPLE_INDEX}.",
    )
    return parser.parse_args()


def load_actions(label_map_path: Path, fallback_actions: Optional[list[str]]) -> np.ndarray:
    """Load action labels in the exact class-index order used during training."""
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        ordered_actions = sorted(label_map, key=label_map.get)
        return np.array(ordered_actions)

    if fallback_actions:
        return np.array(fallback_actions)
    return np.array(ACTIONS)


def build_preprocess_args(args: argparse.Namespace, actions: np.ndarray) -> argparse.Namespace:
    """Create the argument object expected by preprocess_dataset()."""
    return argparse.Namespace(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        actions=list(actions),
        auto_actions=False,
        test_size=args.test_size,
    )


def recreate_lstm_architecture(
    input_shape: tuple[int, int],
    num_classes: int,
) -> Sequential:
    """Recreate the exact training architecture for weights-only loading.

    Use this only when you saved weights separately with model.save_weights().
    The architecture must match training exactly: same layers, same order, same
    units, and same output class count.
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def explain_persistence_methods() -> None:
    """Print the practical difference between model and weight persistence."""
    print("\nPersistence Notes")
    print(
        "model.save() stores the full model: architecture, trained weights, "
        "and compile/training configuration when available."
    )
    print(
        "model.save_weights() stores only learned weights. To use those "
        "weights later, recreate the exact same architecture first, then call "
        "model.load_weights()."
    )
    print(
        "Reloading avoids retraining, which is essential for future inference, "
        "deployment, demos, and reproducible evaluation."
    )


def save_action_model(source_model_path: Path, action_model_path: Path) -> None:
    """Load the trained source model and save it as models/action.h5."""
    if not source_model_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_model_path}")

    action_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Model persistence is important because training can be expensive and the
    # trained weights are the learned gesture-recognition knowledge.
    model = load_model(source_model_path, compile=False)
    model.save(str(action_model_path))
    print(f"Saved full model to: {action_model_path}")

    # Remove the model object from memory to prove the next prediction uses the
    # reloaded file, not the already-loaded Python object.
    del model
    print("Deleted model from memory with: del model")


def validate_model_file(action_model_path: Path) -> None:
    """Check that the persisted model exists before loading."""
    if os.path.exists(action_model_path):
        print("Model file found.")
    else:
        print("Model file not found.")
        raise FileNotFoundError(f"Model file not found: {action_model_path}")


def verify_loaded_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    actions: np.ndarray,
    sample_index: int,
) -> None:
    """Run a prediction and compare it with the ground-truth test label."""
    if len(X_test) == 0:
        raise ValueError("X_test is empty; cannot verify predictions.")

    safe_index = min(max(sample_index, 0), len(X_test) - 1)
    res = model.predict(X_test, verbose=0)

    predicted_label = actions[np.argmax(res[safe_index])]
    actual_label = actions[np.argmax(y_test[safe_index])]

    print(f"\nVerification Sample: {safe_index}")
    print("Predicted:", predicted_label)
    print("Actual:", actual_label)


def run_inference_verification(args: argparse.Namespace) -> None:
    """Save, reload, summarize, and verify the trained model."""
    actions = load_actions(args.label_map_path, args.actions)
    preprocess_args = build_preprocess_args(args, actions)
    _, _, _, X_test, _, y_test = preprocess_dataset(preprocess_args)

    save_action_model(args.source_model_path, args.action_model_path)
    validate_model_file(args.action_model_path)

    # Preferred option: load the complete model in one call.
    model = load_model(args.action_model_path, compile=False)
    print("Model successfully loaded")
    print(f"Number of classes: {len(actions)}")
    model.summary()

    # This helper demonstrates the weights-only path:
    # weights_model = recreate_lstm_architecture(X_test.shape[1:], len(actions))
    # weights_model.load_weights("path/to/weights.weights.h5")
    # Use it when you intentionally save only weights rather than a full model.
    _ = recreate_lstm_architecture(X_test.shape[1:], len(actions))

    verify_loaded_model(model, X_test, y_test, actions, args.sample_index)
    explain_persistence_methods()


def main() -> int:
    """Program entry point."""
    args = parse_args()
    try:
        run_inference_verification(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Inference verification failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
