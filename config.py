"""Repository-wide paths and lightweight runtime helpers.

All scripts should import paths from here instead of assuming a particular
computer, current directory, or developer username.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = DATA_DIR / "models"
CONE_MODEL_PATH = MODELS_DIR / "cone_detection" / "best.pt"
DROWSINESS_MODEL_PATH = MODELS_DIR / "drowsiness" / "shape_predictor_68_face_landmarks.dat"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_runtime_directories() -> None:
    """Create directories used for generated artifacts when needed."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def project_path(value: str | Path) -> Path:
    """Resolve a relative user path from the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
