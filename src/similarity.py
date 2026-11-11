"""Item-item Pearson similarity, using only training ratings.

SQLite (standard library) is used to aggregate co-rated movie pairs without holding
all pair statistics in a large Python dictionary. No ML/recommender library is used.
"""
from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path

from src.config import TRAIN_CSV, SIMILARITIES_CSV, PROCESSED_DIR, ensure_directories

SIMILARITY_FIELDS = ["movieId1", "movieId2", "correlation", "commonUsers"]


def pearson_from_sums(
    n: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float | None:
    """Calculate Pearson correlation from sufficient statistics for co-raters."""
    if n < 2:
        return None
    covariance_numerator = sum_xy - (sum_x * sum_y / n)
    variance_x = sum_x2 - (sum_x * sum_x / n)
    variance_y = sum_y2 - (sum_y * sum_y / n)
    if variance_x <= 1e-12 or variance_y <= 1e-12:
        return None  # Pearson is undefined if either vector is constant.
    denominator = math.sqrt(variance_x * variance_y)
    if denominator <= 1e-12:
        return None
    return max(-1.0, min(1.0, covariance_numerator / denominator))


def compute_similarity_csv(
    train_path: Path = TRAIN_CSV,
    output_path: Path = SIMILARITIES_CSV,
    min_common_users: int = 3,
) -> tuple[Path, int]:
    """Compute item-item Pearson correlations on co-rating users and write CSV."""
    if min_common_users < 2:
        raise ValueError("min_common_users deve ser pelo menos 2.")
    if not train_path.exists():
        raise FileNotFoundError(f"Não existe {train_path}. Execute 'python main.py prepare' primeiro.")
    ensure_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = output_path.with_suffix(".work.sqlite")
    for candidate in (work_path, Path(str(work_path) + "-journal"), Path(str(work_path) + "-wal"), Path(str(work_path) + "-shm")):
        if candidate.exists():
            candidate.unlink()

    connection = sqlite3.connect(str(work_path))
    written = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-50000")  # approximately 50 MiB
        connection.execute(
            "CREATE TABLE ratings (user_id INTEGER NOT NULL, movie_id INTEGER NOT NULL, rating REAL NOT NULL)"
        )
        with train_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            batch = []
            for row in reader:
                batch.append((int(row["userId"]), int(row["movieId"]), float(row["rating"])))
                if len(batch) >= 10000:
                    connection.executemany("INSERT INTO ratings VALUES (?, ?, ?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO ratings VALUES (?, ?, ?)", batch)
        connection.execute("CREATE INDEX idx_ratings_user_movie ON ratings(user_id, movie_id)")
        connection.commit()

        query = """
            SELECT a.movie_id AS movie1,
                   b.movie_id AS movie2,
                   COUNT(*) AS n,
                   SUM(a.rating) AS sum_x,
                   SUM(b.rating) AS sum_y,
                   SUM(a.rating * a.rating) AS sum_x2,
                   SUM(b.rating * b.rating) AS sum_y2,
                   SUM(a.rating * b.rating) AS sum_xy
            FROM ratings AS a
            JOIN ratings AS b
              ON a.user_id = b.user_id
             AND a.movie_id < b.movie_id
            GROUP BY a.movie_id, b.movie_id
            HAVING COUNT(*) >= ?
            ORDER BY a.movie_id, b.movie_id
        """
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=SIMILARITY_FIELDS)
            writer.writeheader()
            cursor = connection.execute(query, (min_common_users,))
            for row in cursor:
                n, sx, sy, sx2, sy2, sxy = row[2:]
                correlation = pearson_from_sums(n, sx, sy, sx2, sy2, sxy)
                if correlation is None:
                    continue
                writer.writerow({
                    "movieId1": int(row[0]),
                    "movieId2": int(row[1]),
                    "correlation": f"{correlation:.10f}",
                    "commonUsers": int(n),
                })
                written += 1
    finally:
        connection.close()
        for candidate in (work_path, Path(str(work_path) + "-journal"), Path(str(work_path) + "-wal"), Path(str(work_path) + "-shm")):
            if candidate.exists():
                candidate.unlink()

    print(f"Similaridades gravadas: {written:,} relações em {output_path} (mínimo de {min_common_users} usuários em comum)")
    return output_path, written
