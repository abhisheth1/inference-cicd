from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"


def list_models():
    return [p.name for p in MODEL_DIR.glob("*")]