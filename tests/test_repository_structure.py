"""Fast checks for portable repository structure and Python syntax."""
import ast

from config import CONE_MODEL_PATH, DROWSINESS_MODEL_PATH, OUTPUTS_DIR, PROJECT_ROOT


def test_central_paths_are_repository_relative():
    assert CONE_MODEL_PATH == PROJECT_ROOT / "data" / "models" / "cone_detection" / "best.pt"
    assert DROWSINESS_MODEL_PATH == PROJECT_ROOT / "data" / "models" / "drowsiness" / "shape_predictor_68_face_landmarks.dat"
    assert OUTPUTS_DIR.parent == PROJECT_ROOT


def test_python_sources_parse():
    python_files = list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py"))
    assert python_files
    for source in python_files:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_required_documentation_exists():
    for relative_path in ("README.md", "requirements.txt", ".gitignore", "data/README.md", "THIRD_PARTY_NOTICES.md"):
        assert (PROJECT_ROOT / relative_path).is_file()
