"""Evaluate a trained sign language recognition model on the test split.

Workflow:
    X_test
    -> model.predict()
    -> probability scores
    -> np.argmax()
    -> predicted class index
    -> actions[class_index]
    -> predicted gesture label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
)
from tensorflow.keras.models import load_model

from preprocess_dataset import ACTIONS, DATA_PATH, SEQUENCE_LENGTH, TEST_SIZE
from preprocess_dataset import preprocess_dataset


DEFAULT_MODEL_PATH = Path("models/best_model.h5")
DEFAULT_LABEL_MAP_PATH = Path("models/label_map.json")
DEFAULT_CONFUSION_MATRIX_PATH = Path("models/confusion_matrix.png")
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SAMPLE_INDEX = 4


def parse_args() -> argparse.Namespace:
    """Parse model evaluation options."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained sign language LSTM model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Trained Keras model path. Default: {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=DEFAULT_LABEL_MAP_PATH,
        help=f"JSON label map from training. Default: {DEFAULT_LABEL_MAP_PATH}.",
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
        help="Fallback action labels if no label map exists.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help=f"Single example index to print first. Default: {DEFAULT_SAMPLE_INDEX}.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of example predictions to print. Default: {DEFAULT_SAMPLE_COUNT}.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_PATH,
        help=f"Output image path. Default: {DEFAULT_CONFUSION_MATRIX_PATH}.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Deprecated: confusion matrices are saved to disk for reliability.",
    )
    return parser.parse_args()


def load_actions(label_map_path: Path, fallback_actions: list[str] | None) -> np.ndarray:
    """Load action labels in the same class-index order used for training."""
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        ordered_labels = sorted(label_map, key=label_map.get)
        return np.array(ordered_labels)

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


def print_single_example(
    predictions: np.ndarray,
    y_test: np.ndarray,
    actions: np.ndarray,
    sample_index: int,
) -> None:
    """Print the requested one-sample prediction demonstration."""
    if len(predictions) == 0:
        print("No test samples available.")
        return

    safe_index = min(max(sample_index, 0), len(predictions) - 1)

    # model.predict() returns one softmax probability vector per sample.
    # np.argmax() selects the most likely class index from that vector.
    predicted_label = actions[np.argmax(predictions[safe_index])]

    # y_test is one-hot encoded, so argmax converts it back to the true class id.
    actual_label = actions[np.argmax(y_test[safe_index])]

    print(f"\nSingle Sample {safe_index}")
    print("Predicted:", predicted_label)
    print("Actual:", actual_label)


def print_prediction_examples(
    yhat: list[int],
    ytrue: list[int],
    actions: np.ndarray,
    sample_count: int,
) -> None:
    """Print several predicted-vs-actual examples."""
    print("\nSample Predictions")
    for index in range(min(sample_count, len(yhat))):
        print(
            f"Sample {index}: "
            f"Predicted = {actions[yhat[index]]}, "
            f"Actual = {actions[ytrue[index]]}"
        )


def save_confusion_matrix(
    ytrue: list[int],
    yhat: list[int],
    actions: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    """Create and save a confusion matrix visualization."""
    # Confusion matrices reveal which gestures are being confused with each
    # other, which is more useful than a single accuracy number for debugging
    # sign language recognition errors.
    matrix = confusion_matrix(ytrue, yhat, labels=np.arange(len(actions)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(actions)),
        yticks=np.arange(len(actions)),
        xticklabels=actions,
        yticklabels=actions,
        title="Sign Language Recognition Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="Actual Label",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                format(matrix[row_index, column_index], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )

    plt.title("Sign Language Recognition Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"\nConfusion matrix saved: {output_path}")

    if show_plot:
        print("Plot display is disabled; open the saved PNG instead.")
    plt.close()


def evaluate_model(args: argparse.Namespace) -> None:
    """Load the model, run predictions, and print evaluation metrics."""
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")

    actions = load_actions(args.label_map_path, args.actions)
    preprocess_args = build_preprocess_args(args, actions)

    _, _, _, X_test, _, y_test = preprocess_dataset(preprocess_args)
    model = load_model(args.model_path, compile=False)

    # model.predict() returns probability scores because the final Dense layer
    # uses softmax. Softmax converts raw outputs into class probabilities.
    predictions = model.predict(X_test, verbose=0)

    # np.argmax() converts probability vectors into class indices by selecting
    # the class with the highest predicted probability.
    yhat = np.argmax(predictions, axis=1).tolist()

    # y_test is one-hot encoded, so np.argmax() converts it back to ground-truth
    # integer class indices for metric calculation.
    ytrue = np.argmax(y_test, axis=1).tolist()

    # Accuracy is the percentage of predictions where predicted class equals
    # the ground-truth class. It is useful, but not sufficient when classes are
    # imbalanced or when specific gesture mix-ups matter more than others.
    accuracy = accuracy_score(ytrue, yhat)
    correct_predictions = int(np.sum(np.array(yhat) == np.array(ytrue)))
    total_predictions = len(ytrue)
    incorrect_predictions = total_predictions - correct_predictions

    print_single_example(predictions, y_test, actions, args.sample_index)
    print_prediction_examples(yhat, ytrue, actions, args.sample_count)

    print("\nMultilabel Confusion Matrix")
    print(multilabel_confusion_matrix(ytrue, yhat))

    print("\nOverall Performance")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Classification Accuracy: {accuracy:.4f}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {incorrect_predictions}")

    print("\nStandard Confusion Matrix")
    print(confusion_matrix(ytrue, yhat, labels=np.arange(len(actions))))

    print("\nClassification Report")
    print(
        classification_report(
            ytrue,
            yhat,
            labels=np.arange(len(actions)),
            target_names=actions,
            zero_division=0,
        )
    )

    save_confusion_matrix(
        ytrue,
        yhat,
        actions,
        args.confusion_matrix_path,
        args.show_plot,
    )

    print("\nInterpretation Notes")
    print("True Positives: samples correctly predicted as a specific gesture.")
    print("True Negatives: samples correctly predicted as not being that gesture.")
    print("False Positives: samples incorrectly predicted as that gesture.")
    print("False Negatives: samples of that gesture missed by the model.")
    print(
        "Confusion matrices help identify which gestures are being mixed up, "
        "which is critical for improving sign language datasets and models."
    )
    print(
        "Accuracy alone can hide weak per-class performance, especially when "
        "some gestures have more samples than others."
    )


def main() -> int:
    """Program entry point."""
    args = parse_args()
    try:
        evaluate_model(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
