"""Configuration loaded from environment variables or a local .env file."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_env_file() -> None:
    """Load a small, dependency-free .env file without overriding real env vars."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def setting(name: str, default: str) -> str:
    return os.getenv(name, default)


NEO4J_URI = setting("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = setting("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = setting("NEO4J_PASSWORD", "moviegraph123")
NEO4J_DATABASE = setting("NEO4J_DATABASE", "neo4j")
MOVIELENS_URL = setting(
    "MOVIELENS_URL",
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
)
MOVIES_CSV = RAW_DIR / "movies.csv"
RATINGS_CSV = RAW_DIR / "ratings.csv"
TRAIN_CSV = PROCESSED_DIR / "ratings_train.csv"
TEST_CSV = PROCESSED_DIR / "ratings_test.csv"
SIMILARITIES_CSV = PROCESSED_DIR / "similarities.csv"
EVALUATION_CSV = PROCESSED_DIR / "evaluation_results.csv"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
