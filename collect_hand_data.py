"""Collect dynamic hand landmark sequences for sign language training.

The dataset layout is:

dataset/
    hello/
        0/
            0.npy
            1.npy
            ...

Each .npy file contains one frame of left-hand and right-hand landmarks:
21 left hand points * 3 coordinates + 21 right hand points * 3 coordinates
= 126 float32 values.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:
    missing_package = exc.name or "a required package"
    print(
        f"Missing dependency: {missing_package}\n\n"
        "Install dependencies in the Python 3.11 virtual environment:\n"
        "  source .venv311/bin/activate\n"
        "  python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

# Dataset configuration.
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
    "again",
    "eat",
    "eat_food",
    "more",
    "go_to",
    "fine",
    "like",
    "learn",
    "name",
    "meet",
    "nice",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
DATA_PATH = Path("dataset")

# Collection behavior.
COUNTDOWN_SECONDS = 3
WINDOW_NAME = "Hand Movement Data Collection - Press q to quit"
DEFAULT_CAMERA_WARMUP_FRAMES = 60
FRAME_RETRY_DELAY_SECONDS = 0.05


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
    """A camera that has opened and produced at least one frame."""

    capture: cv2.VideoCapture
    first_frame: np.ndarray
    config: CameraConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Collect MediaPipe hand landmark sequences into .npy files."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help=f"Root dataset directory. Default: {DATA_PATH}.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Requested camera width. Default: 0 keeps the camera default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Requested camera height. Default: 0 keeps the camera default.",
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
        help="Read attempts before treating the camera as unavailable.",
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
        "--no-flip",
        action="store_true",
        help="Do not mirror the webcam image horizontally.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    """Configure terminal logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def avfoundation_backend() -> Optional[int]:
    """Return OpenCV's AVFoundation backend constant when available."""
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.CAP_AVFOUNDATION
    return None


def backend_name(backend: Optional[int]) -> str:
    """Return a readable backend name for logs."""
    if backend is None:
        return "default"
    if backend == avfoundation_backend():
        return "avfoundation"
    return str(backend)


def camera_config_candidates(args: argparse.Namespace) -> list[CameraConfig]:
    """Build camera configurations to try until one returns frames."""
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
    """Open a webcam handle."""
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
        "Camera opened: index=%s backend=%s width=%s height=%s fps=%s",
        config.camera_index,
        backend_name(config.backend),
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
    """Read one frame from the camera, retrying during warm-up."""
    for attempt in range(max(retries, 1)):
        success, frame = capture.read()
        if success and frame is not None and frame.size:
            return frame
        if attempt == 0:
            logging.debug("Camera read returned no frame; retrying.")
        time.sleep(retry_delay_seconds)
    return None


def open_working_camera(args: argparse.Namespace) -> CameraSession:
    """Open the first camera configuration that returns frames."""
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
            return CameraSession(capture=capture, first_frame=frame, config=config)
        except CameraOpenError as exc:
            errors.append(str(exc))
            release_capture(capture)

    raise CameraReadError(
        "Could not read frames from the webcam after trying these options: "
        + " | ".join(errors)
    )


def release_capture(capture: Optional[cv2.VideoCapture]) -> None:
    """Release webcam resources."""
    if capture is not None and capture.isOpened():
        capture.release()


def create_dataset_folders(data_path: Path) -> None:
    """Create action and sequence folders before recording starts."""
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            (data_path / action / str(sequence)).mkdir(parents=True, exist_ok=True)
    logging.info("Dataset folders are ready under %s", data_path)


def mediapipe_detection(image: np.ndarray, holistic_model) -> object:
    """Run MediaPipe Holistic inference on a BGR frame."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = holistic_model.process(rgb_image)
    rgb_image.flags.writeable = True
    return results


def extract_hand_keypoints(results) -> np.ndarray:
    """Extract left and right hand x/y/z landmarks as a 126-value vector."""
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


def draw_detected_landmarks(image: np.ndarray, results) -> None:
    """Draw hand and pose landmarks, ignoring face landmarks."""
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


def draw_status_overlay(
    image: np.ndarray,
    action: str,
    sequence: int,
    frame_num: int,
    status: str,
    countdown: Optional[int] = None,
) -> None:
    """Render collection progress on the OpenCV window."""
    cv2.rectangle(image, (0, 0), (650, 130), (0, 0, 0), thickness=-1)
    lines = [
        f"Gesture: {action.upper()}",
        f"Sequence: {sequence + 1} / {NO_SEQUENCES}",
        f"Frame: {frame_num + 1} / {SEQUENCE_LENGTH}",
        status if countdown is None else f"{status} {countdown}",
    ]
    colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (80, 255, 80)]

    for index, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(
            image,
            line,
            (12, 28 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )


def show_frame(image: np.ndarray) -> bool:
    """Show a frame and return False when the user presses q."""
    cv2.imshow(WINDOW_NAME, image)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def wait_with_countdown(
    capture: cv2.VideoCapture,
    holistic_model,
    args: argparse.Namespace,
    action: str,
    sequence: int,
    initial_frame: Optional[np.ndarray] = None,
) -> bool:
    """Show a get-ready countdown before recording a sequence."""
    logging.info("Gesture=%s sequence=%s: get ready.", action, sequence)
    countdown_start = time.monotonic()

    while True:
        elapsed = time.monotonic() - countdown_start
        remaining = COUNTDOWN_SECONDS - int(elapsed)
        if elapsed >= COUNTDOWN_SECONDS:
            return True

        if initial_frame is not None:
            frame = initial_frame
            initial_frame = None
        else:
            frame = read_camera_frame(capture, args.camera_warmup_frames)

        if frame is None:
            raise CameraReadError("Could not read a frame during countdown.")
        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        results = mediapipe_detection(frame, holistic_model)
        draw_detected_landmarks(frame, results)
        draw_status_overlay(
            frame,
            action,
            sequence,
            0,
            "Get Ready:",
            countdown=max(remaining, 1),
        )

        if not show_frame(frame):
            return False


def record_sequence(
    capture: cv2.VideoCapture,
    holistic_model,
    args: argparse.Namespace,
    action: str,
    sequence: int,
) -> bool:
    """Record and save one fixed-length gesture sequence."""
    sequence_path = args.data_path / action / str(sequence)
    logging.info(
        "Recording gesture=%s sequence=%s/%s",
        action,
        sequence + 1,
        NO_SEQUENCES,
    )

    for frame_num in range(SEQUENCE_LENGTH):
        frame = read_camera_frame(capture, args.camera_warmup_frames)
        if frame is None:
            raise CameraReadError("Could not read a frame while recording.")
        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        results = mediapipe_detection(frame, holistic_model)
        keypoints = extract_hand_keypoints(results)
        file_path = sequence_path / f"{frame_num}.npy"
        np.save(file_path, keypoints)

        draw_detected_landmarks(frame, results)
        draw_status_overlay(frame, action, sequence, frame_num, "Recording...")

        hands_detected = sum(
            landmark is not None
            for landmark in (results.left_hand_landmarks, results.right_hand_landmarks)
        )
        logging.info(
            "Saved %s shape=%s hands=%s/2",
            file_path,
            keypoints.shape,
            hands_detected,
        )

        if not show_frame(frame):
            return False

    return True


def run_collection(args: argparse.Namespace) -> int:
    """Run the full data collection workflow."""
    create_dataset_folders(args.data_path)
    camera_session = open_working_camera(args)
    capture = camera_session.capture
    mp_holistic = mp.solutions.holistic
    initial_frame: Optional[np.ndarray] = camera_session.first_frame

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as holistic_model:
        try:
            for action in ACTIONS:
                logging.info("Starting gesture: %s", action)
                for sequence in range(NO_SEQUENCES):
                    if not wait_with_countdown(
                        capture,
                        holistic_model,
                        args,
                        action,
                        sequence,
                        initial_frame,
                    ):
                        logging.info("Collection stopped by user.")
                        return 0
                    initial_frame = None

                    if not record_sequence(
                        capture,
                        holistic_model,
                        args,
                        action,
                        sequence,
                    ):
                        logging.info("Collection stopped by user.")
                        return 0

            logging.info("Collection complete.")
            return 0
        finally:
            release_capture(capture)
            cv2.destroyAllWindows()


def main() -> int:
    """Program entry point."""
    args = parse_args()
    configure_logging(args.debug)

    try:
        return run_collection(args)
    except CameraOpenError as exc:
        logging.error("%s", exc)
        logging.error(
            "Allow camera access for Terminal, iTerm, VS Code, or your Python "
            "launcher in System Settings > Privacy & Security > Camera."
        )
        return 1
    except CameraReadError as exc:
        logging.error("%s", exc)
        logging.error(
            "Close other camera apps. If needed, retry with: "
            "python collect_hand_data.py --camera-backend default --width 0 --height 0"
        )
        return 1
    except RuntimeError as exc:
        logging.error("Runtime error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
