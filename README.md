# Yoga Pose Detection Web App

An educational computer-vision project that detects five yoga poses from an uploaded image or webcam stream using MediaPipe body landmarks and a scikit-learn Gradient Boosting classifier.

> **Important:** This is a learning and posture-feedback demo, not medical advice or a substitute for a qualified yoga instructor. Predictions can be wrong, especially with poor lighting, partial bodies, unusual camera angles, or poses outside the five trained classes.

## What the app does

1. Reads an image or webcam frame.
2. Converts the OpenCV BGR frame to RGB for MediaPipe.
3. Extracts 33 human-pose landmarks.
4. Flattens the landmark coordinates into model features.
5. Predicts one of the learned pose classes and displays the model confidence.
6. Draws the landmarks and prediction on the output frame.

The project focuses on Downward Dog, Plank, Tree, Goddess, and Warrior II. It includes both the Flask interface and notebooks covering feature extraction, model comparison, web scraping, and a deep-learning experiment.

## Important reproducibility note

The repository includes a pre-trained `detect_pose.pkl` model so the demo can run after dependencies are installed. The original `Machine Learning Code/coords.csv` currently contains only its header and no training rows. Therefore, the original training workflow cannot be reproduced from the checked-in coordinate dataset alone. Do not present the historical accuracy as a newly re-run benchmark until the training data is restored and the model is retrained.

The serialized model was created with scikit-learn 0.24.2. The runtime requirements use a current scikit-learn range; if the old pickle cannot be loaded in your environment, retrain the classifier from a restored coordinate dataset rather than weakening the dependency checks.

## Run the web app

Use Python 3.10 or 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_project.py
python app.py
```

Open `http://127.0.0.1:5000`. The image-upload flow is the most portable. The webcam flow requires camera permissions and a local machine with an accessible camera.

The app also exposes a lightweight health endpoint at `http://127.0.0.1:5000/health`.

## Notebook environment

For the optional notebook experiments:

```bash
python -m pip install -r requirements-notebooks.txt
jupyter notebook
```

The deep-learning notebook may require a GPU or substantial CPU time. The browser app and the notebook experiments are separate execution paths.

## Validation

The dependency-free structural check confirms that the model, Flask templates, JavaScript, requirements, and notebooks are present and readable:

```bash
python scripts/validate_project.py
```

## Repository structure

```text
.
├── app.py                         # Flask application
├── detect_pose.pkl                # Bundled pre-trained classifier
├── requirements.txt               # Web-app dependencies
├── requirements-notebooks.txt     # Optional notebook dependencies
├── scripts/validate_project.py    # Structural preflight check
├── templates/                     # Flask pages
├── static/                        # CSS, JavaScript, and images
├── Machine Learning Code/         # Landmark and classifier notebooks
├── Deep Learning Code/            # Neural-network notebook and assets
└── Web Scraper/                   # Dataset URL collection notebook/files
```

## Recruiter-facing summary

This project demonstrates an end-to-end computer-vision workflow: MediaPipe landmark extraction, feature engineering, comparison of classical ML classifiers, model serialization, Flask API design, image upload handling, webcam streaming, confidence display, and basic reproducibility checks. It is best presented as an academic machine-learning web application, with the missing original coordinate-training data disclosed clearly.
