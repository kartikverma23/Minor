"""Flask application for educational yoga-pose classification."""
from __future__ import annotations

import base64
import os
import pickle
from pathlib import Path
from typing import Generator

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
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
)

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def annotate_frame(input_frame: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Detect landmarks and annotate one OpenCV BGR frame."""
    output_frame = input_frame.copy()
    label = "Unknown Pose"
    confidence = "0.000"

    # MediaPipe expects RGB while OpenCV reads/captures BGR frames.
    rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

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
        features = pd.DataFrame([row])
        label = str(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence_value = float(np.max(probabilities)) * 100
        confidence = f"{confidence_value:.3f}"
        if confidence_value < 50:
            label = "Unknown Pose"

    cv2.rectangle(output_frame, (0, 0), (250, 60), (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(output_frame, "Class", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(output_frame, label, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
    cv2.putText(output_frame, "Confidence", (150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(output_frame, confidence, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
    return output_frame, label, confidence


def using_image(img_path: Path) -> str:
    input_frame = cv2.imread(str(img_path))
    if input_frame is None:
        raise ValueError("The uploaded file is not a readable image.")
    output_frame, _, _ = annotate_frame(input_frame)
    encoded, buffer = cv2.imencode(".jpg", output_frame)
    if not encoded:
        raise ValueError("The annotated image could not be encoded.")
    return base64.b64encode(buffer).decode("ascii")


def using_webcam() -> Generator[bytes, None, None]:
    camera = cv2.VideoCapture(0)
    try:
        while camera.isOpened():
            status, input_frame = camera.read()
            if not status:
                break
            output_frame, _, _ = annotate_frame(input_frame)
            encoded, buffer = cv2.imencode(".jpg", output_frame)
            if not encoded:
                continue
            frame = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    finally:
        camera.release()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/webcam", methods=["GET"])
def webcam():
    return render_template("webcam.html")


@app.route("/video_capture", methods=["GET"])
def video_capture():
    return Response(using_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return Response("No image file was uploaded.", status=400, mimetype="text/plain")
    if not allowed_file(uploaded.filename):
        return Response("Only PNG and JPEG images are supported.", status=400, mimetype="text/plain")

    filename = secure_filename(uploaded.filename)
    file_path = UPLOAD_DIR / filename
    uploaded.save(file_path)
    try:
        predictions = using_image(file_path)
        return Response(predictions, mimetype="text/plain")
    finally:
        file_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
