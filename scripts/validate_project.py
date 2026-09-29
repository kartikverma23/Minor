"""Run dependency-free structural checks for the yoga pose project."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "app.py",
    "detect_pose.pkl",
    "requirements.txt",
    "templates/index.html",
    "templates/webcam.html",
    "static/js/script.js",
]


def main() -> int:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    notebooks = list(ROOT.rglob("*.ipynb"))
    invalid = []
    for notebook in notebooks:
        try:
            json.loads(notebook.read_text(errors="ignore"))
        except json.JSONDecodeError:
            invalid.append(str(notebook))
    if invalid:
        raise SystemExit(f"Invalid notebooks: {invalid}")
    coords = ROOT / "Machine Learning Code" / "coords.csv"
    coordinate_rows = max(0, sum(1 for _ in coords.open()) - 1) if coords.exists() else 0
    print(f"Required files: {len(REQUIRED_FILES)} present")
    print(f"Valid notebooks: {len(notebooks)}")
    print(f"Training coordinate rows available: {coordinate_rows}")
    if coordinate_rows == 0:
        print("NOTE: coords.csv contains no training rows; the bundled classifier is pre-trained.")
    print("Project structure check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
