"""Real-time sign language landmark detection with MediaPipe Holistic.

This script opens the default webcam, detects pose and hand landmarks in real
time, draws the landmarks on each frame, and exits when the user presses "q".
It is intentionally model-ready: TensorFlow recognition can be added by feeding
extracted landmark keypoints into an LSTM, Transformer, or other classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:
    missing_package = exc.name or "a required package"
    print(
        f"Missing dependency: {missing_package}\n\n"
        "Install the project dependencies in a compatible virtual environment:\n"
        "  python3.11 -m venv .venv311\n"
        "  source .venv311/bin/activate\n"
        "  python -m pip install --upgrade pip\n"
        "  python -m pip install -r requirements.txt\n\n"
        "If python3.11 is not installed on macOS, install it with:\n"
        "  brew install python@3.11",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


WINDOW_NAME = "OpenCV Feed"
DEFAULT_MODEL_PATH = Path("models/action.h5")
DEFAULT_LABEL_MAP_PATH = Path("models/label_map.json")
DEFAULT_PREDICTION_LOG_PATH = Path("Logs/predictions.csv")
DEFAULT_CAMERA_WARMUP_FRAMES = 60
FRAME_RETRY_DELAY_SECONDS = 0.05
STABILIZATION_WINDOW = 10
MAX_SENTENCE_LENGTH = 5


class CameraOpenError(RuntimeError):
    """Raised when OpenCV cannot open the requested webcam."""


class CameraReadError(RuntimeError):
    """Raised when OpenCV opens a webcam but cannot read frames."""


@dataclass(frozen=True)
class CameraConfig:
    """Runtime configuration for webcam capture."""

    camera_index: int
    width: Optional[int]
    height: Optional[int]
    backend: Optional[int]


@dataclass(frozen=True)
class CameraSession:
    """A successfully opened camera and the first readable frame."""

    capture: cv2.VideoCapture
    first_frame: np.ndarray
    config: CameraConfig


@dataclass
class RecognitionState:
    """Runtime state for real-time gesture recognition."""

    sequence: deque[np.ndarray]
    sentence: list[str]
    predictions: deque[int]
    probabilities: np.ndarray
    threshold: float
    accepted_label: str = ""
    confidence: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line options for camera and MediaPipe settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Run real-time pose and hand landmark detection using "
            "MediaPipe Holistic and OpenCV."
        )
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use. Default: 0.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Requested camera frame width. Default: 0 keeps the camera default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Requested camera frame height. Default: 0 keeps the camera default.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "avfoundation", "default"),
        default="auto",
        help="OpenCV camera backend to use on macOS. Default: auto.",
    )
    parser.add_argument(
        "--camera-warmup-frames",
        type=int,
        default=DEFAULT_CAMERA_WARMUP_FRAMES,
        help="Number of read attempts before treating the camera as unavailable.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for initial landmark detection.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for landmark tracking between frames.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="MediaPipe pose model complexity: 0=fast, 1=balanced, 2=accurate.",
    )
    parser.add_argument(
        "--draw-face",
        action="store_true",
        help="Also draw face landmarks. Disabled by default for less clutter.",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Do not mirror the webcam image horizontally.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"TensorFlow/Keras model path. Default: {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=DEFAULT_LABEL_MAP_PATH,
        help=f"JSON label map from training. Default: {DEFAULT_LABEL_MAP_PATH}.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help="Optional fallback text file with one class label per line.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Number of frames to send to the recognition model.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.5,
        help="Minimum model confidence before displaying a class label.",
    )
    parser.add_argument(
        "--prediction-log-path",
        type=Path,
        default=DEFAULT_PREDICTION_LOG_PATH,
        help=f"CSV log path for accepted predictions. Default: {DEFAULT_PREDICTION_LOG_PATH}.",
    )
    parser.add_argument(
        "--disable-threshold-slider",
        action="store_true",
        help="Disable the OpenCV confidence threshold slider.",
    )
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def avfoundation_backend() -> Optional[int]:
    """Return OpenCV's AVFoundation backend constant when it is available."""
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.CAP_AVFOUNDATION
    return None


def backend_name(backend: Optional[int]) -> str:
    """Return a readable backend name for logging."""
    if backend is None:
        return "default"
    if backend == avfoundation_backend():
        return "avfoundation"
    return str(backend)


def camera_config_candidates(args: argparse.Namespace) -> list[CameraConfig]:
    """Build a conservative list of camera configurations to try."""
    requested_size = (args.width or None, args.height or None)
    default_size = (None, None)

    if args.camera_backend == "avfoundation":
        backends = [avfoundation_backend()]
    elif args.camera_backend == "default":
        backends = [None]
    else:
        backends = []
        if platform.system() == "Darwin":
            backends.append(avfoundation_backend())
        backends.append(None)

    configs: list[CameraConfig] = []
    for backend in backends:
        if backend is None and args.camera_backend == "avfoundation":
            continue
        for width, height in (requested_size, default_size):
            config = CameraConfig(args.camera_index, width, height, backend)
            if config not in configs:
                configs.append(config)
    return configs


def open_camera_handle(config: CameraConfig) -> cv2.VideoCapture:
    """Open the webcam handle and raise a clear error if it is unavailable."""
    if config.backend is None:
        capture = cv2.VideoCapture(config.camera_index)
    else:
        capture = cv2.VideoCapture(config.camera_index, config.backend)

    if config.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    if config.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    if not capture.isOpened():
        release_capture(capture)
        raise CameraOpenError(
            "Could not open webcam. "
            f"camera_index={config.camera_index}, "
            f"backend={backend_name(config.backend)}, "
            f"opencv_version={cv2.__version__}, "
            f"python={sys.version.split()[0]}, "
            f"platform={platform.platform()}"
        )

    logging.info(
        "Camera opened: index=%s width=%s height=%s fps=%s",
        config.camera_index,
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        capture.get(cv2.CAP_PROP_FPS),
    )
    return capture


def read_camera_frame(
    capture: cv2.VideoCapture,
    retries: int,
    retry_delay_seconds: float = FRAME_RETRY_DELAY_SECONDS,
) -> Optional[np.ndarray]:
    """Read a frame, retrying to handle slow webcam warm-up."""
    for attempt in range(max(retries, 1)):
        success, frame = capture.read()
        if success and frame is not None and frame.size:
            return frame
        if attempt == 0:
            logging.debug("Camera read returned no frame; retrying.")
        time.sleep(retry_delay_seconds)
    return None


def open_working_camera(args: argparse.Namespace) -> CameraSession:
    """Open the first camera configuration that can actually deliver frames."""
    errors: list[str] = []

    for config in camera_config_candidates(args):
        capture: Optional[cv2.VideoCapture] = None
        try:
            logging.info(
                "Trying camera: index=%s backend=%s width=%s height=%s",
                config.camera_index,
                backend_name(config.backend),
                config.width or "default",
                config.height or "default",
            )
            capture = open_camera_handle(config)
            frame = read_camera_frame(capture, args.camera_warmup_frames)
            if frame is None:
                errors.append(
                    "opened but delivered no frames "
                    f"(backend={backend_name(config.backend)}, "
                    f"width={config.width or 'default'}, "
                    f"height={config.height or 'default'})"
                )
                release_capture(capture)
                continue

            logging.info(
                "Camera is delivering frames: backend=%s shape=%s",
                backend_name(config.backend),
                frame.shape,
            )
            return CameraSession(capture=capture, first_frame=frame, config=config)
        except CameraOpenError as exc:
            errors.append(str(exc))
            release_capture(capture)

    raise CameraReadError(
        "Could not read frames from the webcam after trying these options: "
        + " | ".join(errors)
    )


def release_capture(capture: Optional[cv2.VideoCapture]) -> None:
    """Release OpenCV capture resources safely."""
    if capture is not None and capture.isOpened():
        capture.release()


def mediapipe_detection(image, holistic_model):
    """Run MediaPipe inference on a BGR OpenCV frame."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = holistic_model.process(rgb_image)
    rgb_image.flags.writeable = True
    return results


def extract_keypoints(results) -> np.ndarray:
    """Extract hand landmarks in the same 126-value format used for training."""
    left_hand = (
        np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )
    right_hand = (
        np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )
    return np.concatenate([left_hand, right_hand]).astype(np.float32)


def draw_styled_landmarks(image, results, draw_face: bool = False) -> None:
    """Draw pose, hand, and optionally face landmarks on the frame."""
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
        )

    if draw_face and results.face_landmarks:
        face_connections = getattr(
            mp_holistic,
            "FACEMESH_CONTOURS",
            getattr(mp_holistic, "FACEMESH_TESSELATION", None),
        )
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            face_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )


# Backward-compatible aliases for notebook-style helper names.
process_frame = mediapipe_detection
draw_landmarks = draw_styled_landmarks


def add_status_overlay(
    image: np.ndarray,
    results,
    fps: float,
    recognition_state: RecognitionState,
) -> None:
    """Draw sentence, FPS, hands detected, threshold, and prediction history."""
    hands_detected = sum(
        landmark is not None
        for landmark in (results.left_hand_landmarks, results.right_hand_landmarks)
    )
    sentence = " ".join(recognition_state.sentence).upper()
    header = sentence or "..."
    confidence = int(recognition_state.confidence * 100)
    accepted = (
        f"{recognition_state.accepted_label.upper()} ({confidence}%)"
        if recognition_state.accepted_label
        else "Waiting for stable prediction"
    )

    cv2.rectangle(image, (0, 0), (image.shape[1], 92), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        header,
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{accepted} | FPS: {fps:.0f} | Hands: {hands_detected}/2 | Threshold: {recognition_state.threshold:.2f}",
        (12, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )


def load_labels(labels_path: Optional[Path]) -> Optional[list[str]]:
    """Load class labels from a newline-delimited text file."""
    if labels_path is None:
        return None
    if not labels_path.exists():
        raise RuntimeError(f"Labels file does not exist: {labels_path}")

    labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not labels:
        raise RuntimeError(f"Labels file is empty: {labels_path}")
    return labels


def load_actions(label_map_path: Path, labels_path: Optional[Path]) -> np.ndarray:
    """Load action labels in the same class-index order used during training."""
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        actions = sorted(label_map, key=label_map.get)
        return np.array(actions)

    labels = load_labels(labels_path)
    if labels:
        return np.array(labels)

    raise RuntimeError(
        "No labels found. Provide models/label_map.json or pass --labels-path."
    )


def load_tensorflow_model(model_path: Path) -> Any:
    """Load the trained TensorFlow/Keras LSTM model."""
    if not model_path.exists():
        raise RuntimeError(f"Model path does not exist: {model_path}")

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required when --model-path is provided. "
            "Install it with: python3 -m pip install tensorflow"
        ) from exc

    logging.info("Loading TensorFlow model from %s", model_path)
    return load_model(model_path, compile=False)


def generate_colors(num_classes: int) -> list[tuple[int, int, int]]:
    """Generate stable BGR colors for probability bars."""
    base_colors = [
        (245, 117, 16),
        (117, 245, 16),
        (16, 117, 245),
        (245, 16, 117),
        (117, 16, 245),
        (16, 245, 117),
    ]
    return [base_colors[index % len(base_colors)] for index in range(num_classes)]


def prob_viz(
    probabilities: np.ndarray,
    actions: np.ndarray,
    image: np.ndarray,
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    """Draw horizontal probability bars, labels, and confidence scores."""
    start_y = 104
    bar_x = 12
    label_x = 22
    percent_x = 275
    max_bar_width = 250
    row_height = 32

    for index, probability in enumerate(probabilities):
        y1 = start_y + index * row_height
        y2 = y1 + 24
        bar_width = int(max_bar_width * float(probability))

        cv2.rectangle(image, (bar_x, y1), (bar_x + max_bar_width, y2), (35, 35, 35), -1)
        cv2.rectangle(image, (bar_x, y1), (bar_x + bar_width, y2), colors[index], -1)
        cv2.putText(
            image,
            actions[index],
            (label_x, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{probability * 100:.0f}%",
            (percent_x, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def draw_prediction_history(
    image: np.ndarray,
    prediction_history: deque[int],
    actions: np.ndarray,
) -> None:
    """Draw the last 10 prediction labels on the right side of the frame."""
    panel_width = 240
    x1 = max(image.shape[1] - panel_width, 0)
    cv2.rectangle(image, (x1, 104), (image.shape[1], 448), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        "Last 10 Predictions",
        (x1 + 12, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for row, prediction_index in enumerate(list(prediction_history)[-10:]):
        cv2.putText(
            image,
            actions[prediction_index],
            (x1 + 12, 164 + row * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )


def ensure_prediction_log(log_path: Path) -> None:
    """Create the prediction log file with a header if needed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "prediction", "confidence"])


def log_prediction(log_path: Path, prediction: str, confidence: float) -> None:
    """Append an accepted prediction to Logs/predictions.csv."""
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                prediction,
                f"{confidence:.6f}",
            ]
        )


def is_stable_prediction(predictions: deque[int], predicted_index: int) -> bool:
    """Return True when the last 10 predictions agree with the current class."""
    if len(predictions) < STABILIZATION_WINDOW:
        return False
    recent_predictions = list(predictions)[-STABILIZATION_WINDOW:]
    return len(set(recent_predictions)) == 1 and recent_predictions[0] == predicted_index


def update_recognition(
    state: RecognitionState,
    model: Any,
    actions: np.ndarray,
    keypoints: np.ndarray,
    sequence_length: int,
    log_path: Path,
) -> None:
    """Append keypoints, run LSTM inference, smooth predictions, and update text."""
    state.sequence.append(keypoints)
    if len(state.sequence) < sequence_length:
        return

    input_sequence = np.expand_dims(np.array(state.sequence), axis=0)
    probabilities = model.predict(input_sequence, verbose=0)[0]
    state.probabilities = probabilities
    prediction_index = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction_index])
    state.predictions.append(prediction_index)
    state.confidence = confidence

    if is_stable_prediction(state.predictions, prediction_index) and confidence > state.threshold:
        predicted_action = str(actions[prediction_index])
        if not state.sentence or predicted_action != state.sentence[-1]:
            state.sentence.append(predicted_action)
            state.sentence = state.sentence[-MAX_SENTENCE_LENGTH:]
            log_prediction(log_path, predicted_action, confidence)
        state.accepted_label = predicted_action


def validate_model_output(model: Any, actions: np.ndarray) -> None:
    """Ensure the loaded model output matches the available action labels."""
    output_shape = getattr(model, "output_shape", None)
    if output_shape is None or output_shape[-1] is None:
        return

    model_classes = int(output_shape[-1])
    if model_classes != len(actions):
        raise RuntimeError(
            "Model/action mismatch: "
            f"model outputs {model_classes} classes but {len(actions)} labels were loaded."
        )


def validate_keypoints_for_model(model: Any, keypoints: np.ndarray) -> None:
    """Ensure live keypoints match the model's expected feature width."""
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None or input_shape[-1] is None:
        return

    expected_features = int(input_shape[-1])
    if keypoints.shape[0] != expected_features:
        raise RuntimeError(
            "Live keypoint/model mismatch: "
            f"model expects {expected_features} features but extraction returned "
            f"{keypoints.shape[0]}. Use the same keypoint extractor for collection, "
            "training, and live inference."
        )


def create_threshold_slider(initial_threshold: float, disabled: bool) -> None:
    """Create an OpenCV trackbar for real-time confidence-threshold tuning."""
    cv2.namedWindow(WINDOW_NAME)
    if disabled:
        return
    cv2.createTrackbar(
        "Threshold %",
        WINDOW_NAME,
        int(max(0.0, min(initial_threshold, 1.0)) * 100),
        100,
        lambda _: None,
    )


def read_threshold_slider(current_threshold: float, disabled: bool) -> float:
    """Read the OpenCV threshold slider as a 0.0-1.0 value."""
    if disabled:
        return current_threshold
    return cv2.getTrackbarPos("Threshold %", WINDOW_NAME) / 100.0


def run_webcam_loop(args: argparse.Namespace) -> int:
    """Run the real-time webcam processing loop."""
    model = load_tensorflow_model(args.model_path)
    actions = load_actions(args.label_map_path, args.labels_path)
    validate_model_output(model, actions)
    ensure_prediction_log(args.prediction_log_path)
    recognition_state = RecognitionState(
        sequence=deque(maxlen=args.sequence_length),
        sentence=[],
        predictions=deque(maxlen=STABILIZATION_WINDOW),
        probabilities=np.zeros(len(actions), dtype=np.float32),
        threshold=args.prediction_threshold,
    )
    colors = generate_colors(len(actions))

    camera_session = open_working_camera(args)
    capture = camera_session.capture
    mp_holistic = mp.solutions.holistic
    previous_time = time.perf_counter()
    pending_frame: Optional[np.ndarray] = camera_session.first_frame
    keypoint_shape_checked = False
    create_threshold_slider(args.prediction_threshold, args.disable_threshold_slider)

    # MediaPipe Holistic tracks pose and hand landmarks across video frames.
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=args.draw_face,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as holistic_model:
        try:
            while capture.isOpened():
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                else:
                    frame = read_camera_frame(capture, args.camera_warmup_frames)

                if frame is None:
                    logging.warning(
                        "Could not read a frame from the webcam after %s attempts.",
                        args.camera_warmup_frames,
                    )
                    break

                if not args.no_flip:
                    frame = cv2.flip(frame, 1)

                results = mediapipe_detection(frame, holistic_model)
                draw_styled_landmarks(frame, results, draw_face=args.draw_face)
                keypoints = extract_keypoints(results)
                if not keypoint_shape_checked:
                    validate_keypoints_for_model(model, keypoints)
                    keypoint_shape_checked = True

                recognition_state.threshold = read_threshold_slider(
                    recognition_state.threshold,
                    args.disable_threshold_slider,
                )
                update_recognition(
                    recognition_state,
                    model,
                    actions,
                    keypoints,
                    args.sequence_length,
                    args.prediction_log_path,
                )

                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                add_status_overlay(frame, results, fps, recognition_state)
                prob_viz(recognition_state.probabilities, actions, frame, colors)
                draw_prediction_history(frame, recognition_state.predictions, actions)

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    logging.info("Exit requested with q key.")
                    break
        finally:
            release_capture(capture)
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    """Program entry point."""
    args = parse_args()
    configure_logging(args.debug)

    try:
        return run_webcam_loop(args)
    except CameraOpenError as exc:
        logging.error("%s", exc)
        logging.error(
            "macOS troubleshooting: allow camera access for Terminal, iTerm, "
            "VS Code, or your Python launcher in System Settings > Privacy & "
            "Security > Camera. Also verify no other app is using the webcam."
        )
        return 1
    except CameraReadError as exc:
        logging.error("%s", exc)
        logging.error(
            "macOS troubleshooting: close FaceTime, Zoom, Chrome, or any app "
            "using the camera. If this still fails, try: "
            "python sign_language.py --camera-backend default --width 0 --height 0"
        )
        return 1
    except RuntimeError as exc:
        logging.error("Runtime error: %s", exc)
        logging.error(
            "If this mentions NSOpenGLPixelFormat, kGpuService, or display "
            "creation, run the script from a normal macOS Terminal session "
            "with access to the desktop instead of a headless environment."
        )
        return 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
