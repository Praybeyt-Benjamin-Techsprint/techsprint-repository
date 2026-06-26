"""Preprocess collected sign language sequences for TensorFlow training.

This script reads a dataset organized as:

dataset/
    hello/
        0/
            0.npy
            1.npy
            ...

It produces:
    X: sequence features shaped as (samples, sequence_length, keypoints)
    y: one-hot labels shaped as (samples, number_of_actions)
    X_train, X_test, y_train, y_test: train/test partitions

    
"""



from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# Preprocessing configuration.
DATA_PATH = Path("dataset")
ACTIONS = [
    "hello",
    "thank_you",
    "see_you_later",
    "see",
    "you",
    "later",
    "yes",
    "no",
    "help",
    "me",
    "father",
    "mother",
    "abuse",
    "please",
    "want",
    "what",
    "eat_food",
    "more",
    "go_to",
    "fine",
    "like",
    "name",
    "meet",
    "nice",
    "Sorry", 
    "where",
    "call",
]
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line options for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Load sign language .npy sequences and create X/y arrays."
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
        help=f"Expected frames per sequence. Default: {SEQUENCE_LENGTH}.",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=ACTIONS,
        help="Action labels to load. Defaults to the configured ACTIONS list.",
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
        help=f"Fraction of samples used for testing. Default: {TEST_SIZE}.",
    )
    return parser.parse_args()


def discover_actions(data_path: Path) -> list[str]:
    """Return sorted action folder names from DATA_PATH."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    actions = sorted(
        entry.name
        for entry in data_path.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    if not actions:
        raise ValueError(f"No action folders found in dataset path: {data_path}")
    return actions


def create_label_map(actions: Iterable[str]) -> dict[str, int]:
    """Assign each action label a unique integer class id."""
    return {label: index for index, label in enumerate(actions)}


def numeric_sort_key(path: Path) -> tuple[int, str]:
    """Sort numeric sequence folders and frame files in natural order."""
    try:
        return int(path.stem), path.stem
    except ValueError:
        return (10**9, path.stem)


def load_sequence(
    sequence_path: Path,
    sequence_length: int,
    expected_feature_size: Optional[int],
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Load one sequence folder into a fixed-length frame window."""
    window = []

    for frame_num in range(sequence_length):
        frame_path = sequence_path / f"{frame_num}.npy"
        if not frame_path.exists():
            print(f"Skipping incomplete sequence, missing file: {frame_path}")
            return None, expected_feature_size

        try:
            frame_keypoints = np.load(frame_path)
        except OSError as exc:
            print(f"Skipping unreadable file: {frame_path} ({exc})")
            return None, expected_feature_size

        frame_keypoints = np.asarray(frame_keypoints, dtype=np.float32).reshape(-1)
        if expected_feature_size is None:
            expected_feature_size = frame_keypoints.shape[0]
        elif frame_keypoints.shape[0] != expected_feature_size:
            print(
                "Skipping sequence with unexpected feature size: "
                f"{frame_path} has {frame_keypoints.shape[0]}, "
                f"expected {expected_feature_size}"
            )
            return None, expected_feature_size

        window.append(frame_keypoints)

    return np.array(window, dtype=np.float32), expected_feature_size


def load_dataset(
    data_path: Path,
    actions: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load all valid sequences and integer labels from the dataset."""
    label_map = create_label_map(actions)
    sequences = []
    labels = []
    expected_feature_size: Optional[int] = None

    print(f"Label map: {label_map}")

    for action in actions:
        action_path = data_path / action
        if not action_path.exists():
            print(f"Skipping missing action folder: {action_path}")
            continue

        sequence_paths = sorted(
            (path for path in action_path.iterdir() if path.is_dir()),
            key=numeric_sort_key,
        )
        if not sequence_paths:
            print(f"No sequence folders found for action: {action}")
            continue

        for sequence_path in sequence_paths:
            window, expected_feature_size = load_sequence(
                sequence_path,
                sequence_length,
                expected_feature_size,
            )
            if window is None:
                continue

            sequences.append(window)
            labels.append(label_map[action])

    if not sequences:
        raise ValueError("No valid sequences were loaded from the dataset.")

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        label_map,
    )


def can_stratify(labels: np.ndarray) -> bool:
    """Return True when every class has enough samples for stratified split."""
    _, counts = np.unique(labels, return_counts=True)
    return bool(np.all(counts >= 2))


def preprocess_dataset(args: argparse.Namespace) -> tuple[np.ndarray, ...]:
    """Load features, encode labels, split data, and print array shapes."""
    data_path = Path(os.fspath(args.data_path))
    actions = discover_actions(data_path) if args.auto_actions else list(args.actions)

    # Label encoding converts class names like "hello" into integers.
    X, label_ids, label_map = load_dataset(data_path, actions, args.sequence_length)

    # One-hot encoding turns integer labels into vectors for softmax training.
    y = to_categorical(label_ids, num_classes=len(label_map))

    stratify = label_ids if can_stratify(label_ids) else None
    if stratify is None:
        print("Warning: not enough samples per class for stratified splitting.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X, y, X_train, X_test, y_train, y_test


def main() -> int:
    """Program entry point."""
    args = parse_args()
    try:
        preprocess_dataset(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Preprocessing failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
