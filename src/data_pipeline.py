"""MovieLens downloading, CSV validation, and deterministic 80/20 split."""
from __future__ import annotations

import csv
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from src.config import MOVIELENS_URL, MOVIES_CSV, RATINGS_CSV, RAW_DIR, PROCESSED_DIR, TRAIN_CSV, TEST_CSV, ensure_directories

RATING_FIELDS = ["userId", "movieId", "rating", "timestamp"]
MOVIE_FIELDS = ["movieId", "title", "genres"]


def download_dataset(force: bool = False) -> tuple[Path, Path]:
    """Download official MovieLens latest-small and extract only the required CSVs."""
    ensure_directories()
    if not force and MOVIES_CSV.exists() and RATINGS_CSV.exists():
        print(f"MovieLens já existe em {RAW_DIR}")
        return MOVIES_CSV, RATINGS_CSV

    archive_path = RAW_DIR / "ml-latest-small.zip"
    request = urllib.request.Request(
        MOVIELENS_URL,
        headers={"User-Agent": "movie-recommender-course-project/1.0"},
    )
    print(f"Baixando dataset oficial: {MOVIELENS_URL}")
    try:
        with urllib.request.urlopen(request, timeout=90) as response, archive_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível baixar o MovieLens. Verifique sua conexão ou baixe "
            "ml-latest-small.zip em https://grouplens.org/datasets/movielens/ e coloque-o em data/raw/."
        ) from exc

    wanted = {"movies.csv": MOVIES_CSV, "ratings.csv": RATINGS_CSV}
    found: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                basename = Path(member).name
                if basename not in wanted:
                    continue
                with archive.open(member) as source, wanted[basename].open("wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                found.add(basename)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"O arquivo baixado não é um ZIP válido: {archive_path}") from exc

    missing = set(wanted) - found
    if missing:
        raise RuntimeError(f"O ZIP do MovieLens não contém os arquivos esperados: {sorted(missing)}")
    validate_csvs()
    print(f"Dataset preparado em {RAW_DIR}")
    return MOVIES_CSV, RATINGS_CSV


def validate_csvs() -> None:
    for path, required in ((MOVIES_CSV, MOVIE_FIELDS), (RATINGS_CSV, RATING_FIELDS)):
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            missing = set(required) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Colunas ausentes em {path.name}: {sorted(missing)}")
            if next(reader, None) is None:
                raise ValueError(f"O arquivo {path.name} está vazio.")


def read_ratings(path: Path = RATINGS_CSV) -> list[dict[str, int | float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = set(RATING_FIELDS)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Formato inválido em {path}; esperadas as colunas {RATING_FIELDS}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rating = float(row["rating"])
                if not 0.5 <= rating <= 5.0:
                    raise ValueError("rating fora do intervalo [0.5, 5.0]")
                rows.append({
                    "userId": int(row["userId"]),
                    "movieId": int(row["movieId"]),
                    "rating": rating,
                    "timestamp": int(row["timestamp"]),
                })
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Linha inválida {line_number} em {path}: {row}") from exc
    return rows


def write_ratings(path: Path, rows: Iterable[dict[str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RATING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(test_size: float = 0.2, seed: int = 42) -> tuple[Path, Path]:
    """Create a deterministic global random split; the graph receives only train rows."""
    ensure_directories()
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size precisa estar entre 0 e 1, exclusive.")
    validate_csvs()
    ratings = read_ratings(RATINGS_CSV)
    if len(ratings) < 2:
        raise ValueError("O dataset precisa ter pelo menos duas avaliações para separar treino e teste.")
    random.Random(seed).shuffle(ratings)
    split_index = int((1.0 - test_size) * len(ratings))
    split_index = min(max(split_index, 1), len(ratings) - 1)
    train, test = ratings[:split_index], ratings[split_index:]
    write_ratings(TRAIN_CSV, train)
    write_ratings(TEST_CSV, test)
    print(
        f"Split salvo: treino={len(train):,} ({len(train)/len(ratings):.1%}), "
        f"teste={len(test):,} ({len(test)/len(ratings):.1%}), seed={seed}"
    )
    return TRAIN_CSV, TEST_CSV
