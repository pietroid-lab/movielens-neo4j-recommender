"""Neo4j import and graph queries."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

from src.config import (
    MOVIES_CSV, TRAIN_CSV, SIMILARITIES_CSV,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
)

CONSTRAINTS = (
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
    "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
)

COUNT_COMEDIES_QUERY = """
MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)
WHERE r.rating >= $minRating
  AND 'Comedy' IN split(m.genres, '|')
RETURN count(DISTINCT m) AS totalComedies
"""


def _batches(rows: Iterable[dict[str, Any]], batch_size: int):
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _csv_rows(path: Path, transform: Callable[[dict[str, str]], dict[str, Any]]):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            yield transform(row)


class Neo4jStore:
    def __init__(self) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Driver ausente. Execute: pip install -r requirements.txt") from exc
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        self.database = NEO4J_DATABASE
        try:
            self.driver.verify_connectivity()
        except Exception:
            self.driver.close()
            raise

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def execute(self, query: str, **parameters: Any):
        return self.driver.execute_query(query, database_=self.database, **parameters)

    def prepare_schema(self, reset_graph: bool = False) -> None:
        if reset_graph:
            print("Limpando o banco configurado (todos os nós e relacionamentos)...")
            self.execute("MATCH (n) DETACH DELETE n")
        for query in CONSTRAINTS:
            self.execute(query)

    def import_graph(
        self,
        movies_path: Path = MOVIES_CSV,
        train_path: Path = TRAIN_CSV,
        similarities_path: Path = SIMILARITIES_CSV,
        reset_graph: bool = False,
        batch_size: int = 1000,
    ) -> None:
        for path in (movies_path, train_path, similarities_path):
            if not path.exists():
                raise FileNotFoundError(f"Arquivo necessário não encontrado: {path}")
        self.prepare_schema(reset_graph=reset_graph)

        movie_query = """
        UNWIND $rows AS row
        MERGE (m:Movie {movieId: row.movieId})
        SET m.title = row.title, m.genres = row.genres
        """
        movies = _csv_rows(movies_path, lambda row: {
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
        })
        total_movies = 0
        for batch in _batches(movies, batch_size):
            self.execute(movie_query, rows=batch)
            total_movies += len(batch)
        print(f"Filmes importados/atualizados: {total_movies:,}")

        rating_query = """
        UNWIND $rows AS row
        MATCH (m:Movie {movieId: row.movieId})
        MERGE (u:User {userId: row.userId})
        MERGE (u)-[r:RATED]->(m)
        SET r.rating = row.rating, r.timestamp = row.timestamp
        """
        ratings = _csv_rows(train_path, lambda row: {
            "userId": int(row["userId"]),
            "movieId": int(row["movieId"]),
            "rating": float(row["rating"]),
            "timestamp": int(row["timestamp"]),
        })
        total_ratings = 0
        for batch in _batches(ratings, batch_size):
            self.execute(rating_query, rows=batch)
            total_ratings += len(batch)
        print(f"Avaliações de treino importadas/atualizadas: {total_ratings:,}")

        similarity_query = """
        UNWIND $rows AS row
        MATCH (m1:Movie {movieId: row.movieId1})
        MATCH (m2:Movie {movieId: row.movieId2})
        MERGE (m1)-[s:SIMILAR]->(m2)
        SET s.correlation = row.correlation,
            s.commonUsers = row.commonUsers
        """
        similarities = _csv_rows(similarities_path, lambda row: {
            "movieId1": int(row["movieId1"]),
            "movieId2": int(row["movieId2"]),
            "correlation": float(row["correlation"]),
            "commonUsers": int(row["commonUsers"]),
        })
        total_similarities = 0
        for batch in _batches(similarities, batch_size):
            self.execute(similarity_query, rows=batch)
            total_similarities += len(batch)
        print(f"Relacionamentos SIMILAR importados/atualizados: {total_similarities:,}")

        records, _, _ = self.execute("""
        MATCH (u:User) WITH count(u) AS users
        MATCH (m:Movie) WITH users, count(m) AS movies
        MATCH ()-[r:RATED]->() WITH users, movies, count(r) AS ratings
        MATCH ()-[s:SIMILAR]->() RETURN users, movies, ratings, count(s) AS similarities
        """)
        if records:
            print("Resumo do grafo:", records[0].data())

    def count_comedies(self, user_id: int, min_rating: float = 4.0) -> int:
        records, _, _ = self.execute(COUNT_COMEDIES_QUERY, userId=user_id, minRating=min_rating)
        return int(records[0]["totalComedies"]) if records else 0
