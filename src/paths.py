from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"


def ensure_dirs() -> None:
    for directory in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
