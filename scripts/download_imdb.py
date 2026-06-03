from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMDB_ARCHIVE, IMDB_EXTRACTED_DIR, IMDB_URL, RAW_DIR, ensure_directories


def download_file(url: str, output_path: Path) -> None:
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as target:
        shutil.copyfileobj(response, target)


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    print(f"Extracting: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(output_dir)


def main() -> None:
    ensure_directories()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not IMDB_ARCHIVE.exists():
        download_file(IMDB_URL, IMDB_ARCHIVE)
    else:
        print(f"Archive already exists: {IMDB_ARCHIVE}")

    if IMDB_EXTRACTED_DIR.exists():
        print(f"Dataset already extracted: {IMDB_EXTRACTED_DIR}")
        return

    extract_archive(IMDB_ARCHIVE, RAW_DIR)
    extracted_source = RAW_DIR / "aclImdb"
    if not extracted_source.exists():
        raise FileNotFoundError("Extraction completed but aclImdb directory was not found.")
    print(f"Dataset ready: {IMDB_EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
