"""Flask application for educational yoga-pose classification."""
from __future__ import annotations

import base64
import os
import pickle
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "detect_pose.pkl"
UPLOAD_DIR = BASE_DIR / "upload"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD_DIR.mkdir(exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = None
pose_init_error = None


def get_pose():
    """Create the Python MediaPipe graph only when image inference needs it."""
    global pose, pose_init_error
    if pose is not None:
        return pose
    if pose_init_error is not None:
        raise RuntimeError(pose_init_error)
    try:
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        )
        return pose
    except Exception as error:
        pose_init_error = (
            "Python MediaPipe could not initialize on this host. "
            "Use the browser webcam demo or run on a host with MediaPipe graphics support. "
            f"Details: {error}"
        )
        raise RuntimeError(pose_init_error) from error


def load_classifier():
    """Load the checked-in scikit-learn 0.24 model on a compatible runtime.

    scikit-learn 1.2 keeps the old tree layout, but its GradientBoostingClassifier
    expects the fitted loss object under ``_loss``. Older pickles store it under
    ``loss_`` in the serialized state, so restore that state after unpickling.
    """
    with MODEL_PATH.open("rb") as model_file:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipeline = pickle.load(model_file)

    for _, step in getattr(pipeline, "steps", []):
        state = getattr(step, "__dict__", {})
        if "_loss" not in state and "loss_" in state:
            step._loss = state["loss_"]
    return pipeline


model = load_classifier()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_landmarks(row: list[float]) -> tuple[str, str]:
    """Classify one flattened list of 33 (x, y, z) landmarks."""
    if len(row) != 99:
        raise ValueError("Expected 99 landmark coordinates.")
    features = pd.DataFrame([row])
    label = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence_value = float(np.max(probabilities)) * 100
    confidence = f"{confidence_value:.3f}"
    if confidence_value < 50:
        label = "Unknown Pose"
    return label, confidence


def annotate_frame(input_frame: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Detect landmarks and annotate one OpenCV BGR frame."""
    output_frame = input_frame.copy()
    label = "Unknown Pose"
    confidence = "0.000"

    # MediaPipe expects RGB while OpenCV reads/captures BGR frames.
    rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    result = get_pose().process(rgb_frame)

    if result.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=output_frame,
            landmark_list=result.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
        )
        row = [
            value
            for landmark in result.pose_landmarks.landmark
            for value in (landmark.x, landmark.y, landmark.z)
        ]
        label, confidence = classify_landmarks(row)

    cv2.rectangle(output_frame, (0, 0), (290, 60), (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(output_frame, "Class", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(output_frame, label, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
    cv2.putText(output_frame, "Confidence", (175, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(output_frame, confidence, (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
    return output_frame, label, confidence


def infer_image(img_path: Path) -> tuple[str, str, str]:
    input_frame = cv2.imread(str(img_path))
    if input_frame is None:
        raise ValueError("The uploaded file is not a readable image.")
    output_frame, label, confidence = annotate_frame(input_frame)
    encoded, buffer = cv2.imencode(".jpg", output_frame)
    if not encoded:
        raise ValueError("The annotated image could not be encoded.")
    return base64.b64encode(buffer).decode("ascii"), label, confidence


def save_uploaded_image():
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        raise ValueError("No image file was uploaded.")
    if not allowed_file(uploaded.filename):
        raise ValueError("Only PNG and JPEG images are supported.")

    filename = secure_filename(uploaded.filename)
    file_path = UPLOAD_DIR / filename
    uploaded.save(file_path)
    return file_path


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/webcam", methods=["GET"])
def webcam():
    return render_template("webcam.html")


@app.route("/video_capture", methods=["GET"])
def video_capture():
    """Explain why the old server-camera endpoint is no longer used."""
    return Response(
        "The webcam now runs in the browser. Open /webcam and allow camera access.",
        status=410,
        mimetype="text/plain",
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file_path = save_uploaded_image()
    except ValueError as error:
        return Response(str(error), status=400, mimetype="text/plain")
    try:
        encoded, _, _ = infer_image(file_path)
        return Response(encoded, mimetype="text/plain")
    except RuntimeError as error:
        return Response(str(error), status=503, mimetype="text/plain")
    finally:
        file_path.unlink(missing_ok=True)


@app.route("/predict_json", methods=["POST"])
def predict_json():
    """Return an annotated frame and structured result for image uploads."""
    try:
        file_path = save_uploaded_image()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    try:
        encoded, label, confidence = infer_image(file_path)
        return jsonify({"image": encoded, "label": label, "confidence": confidence})
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    finally:
        file_path.unlink(missing_ok=True)


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """Classify 33 landmarks detected in the browser by MediaPipe."""
    payload = request.get_json(silent=True) or {}
    landmarks = payload.get("landmarks")
    if not isinstance(landmarks, list):
        return jsonify({"error": "The request must include a landmarks list."}), 400
    try:
        label, confidence = classify_landmarks([float(value) for value in landmarks])
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
