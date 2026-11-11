"""End-to-end smoke test against a real Neo4j instance.

Run with RUN_NEO4J_INTEGRATION=1 after starting Neo4j.
"""
from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.evaluation import evaluate_model
from src.graph_store import Neo4jStore
from src.recommender import recommend_for_user


@unittest.skipUnless(
    os.getenv("RUN_NEO4J_INTEGRATION") == "1",
    "Defina RUN_NEO4J_INTEGRATION=1 para executar os testes com Neo4j.",
)
class Neo4jEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = Neo4jStore()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.store.close()

    @staticmethod
    def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_import_query_recommend_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            movies_path = root / "movies.csv"
            train_path = root / "ratings_train.csv"
            similarities_path = root / "similarities.csv"
            test_path = root / "ratings_test.csv"
            results_path = root / "evaluation.csv"

            self.write_csv(
                movies_path,
                ["movieId", "title", "genres"],
                [
                    {"movieId": 1, "title": "Comedy One (2000)", "genres": "Comedy"},
                    {"movieId": 2, "title": "Comedy Two (2001)", "genres": "Comedy|Romance"},
                    {"movieId": 3, "title": "Target Drama (2002)", "genres": "Drama"},
                ],
            )
            self.write_csv(
                train_path,
                ["userId", "movieId", "rating", "timestamp"],
                [
                    {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 100},
                    {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 101},
                    {"userId": 2, "movieId": 1, "rating": 5.0, "timestamp": 200},
                    {"userId": 2, "movieId": 2, "rating": 4.0, "timestamp": 201},
                    {"userId": 2, "movieId": 3, "rating": 4.0, "timestamp": 202},
                    {"userId": 3, "movieId": 1, "rating": 4.0, "timestamp": 300},
                    {"userId": 3, "movieId": 2, "rating": 3.0, "timestamp": 301},
                    {"userId": 3, "movieId": 3, "rating": 3.0, "timestamp": 302},
                    {"userId": 4, "movieId": 1, "rating": 3.0, "timestamp": 400},
                    {"userId": 4, "movieId": 2, "rating": 2.0, "timestamp": 401},
                    {"userId": 4, "movieId": 3, "rating": 2.0, "timestamp": 402},
                ],
            )
            self.write_csv(
                similarities_path,
                ["movieId1", "movieId2", "correlation", "commonUsers"],
                [
                    {"movieId1": 1, "movieId2": 2, "correlation": 1.0, "commonUsers": 4},
                    {"movieId1": 1, "movieId2": 3, "correlation": 1.0, "commonUsers": 3},
                    {"movieId1": 2, "movieId2": 3, "correlation": 1.0, "commonUsers": 3},
                ],
            )
            self.write_csv(
                test_path,
                ["userId", "movieId", "rating", "timestamp"],
                [{"userId": 1, "movieId": 3, "rating": 5.0, "timestamp": 500}],
            )

            self.store.import_graph(
                movies_path=movies_path,
                train_path=train_path,
                similarities_path=similarities_path,
                reset_graph=True,
            )

            self.assertEqual(self.store.count_comedies(user_id=1, min_rating=4.0), 2)

            recommendations = recommend_for_user(self.store, user_id=1, k=5, limit=1)
            self.assertEqual(len(recommendations), 1)
            self.assertEqual(recommendations[0]["movieId"], 3)
            self.assertAlmostEqual(recommendations[0]["predictedRating"], 4.5, places=5)

            metrics = evaluate_model(
                self.store,
                test_path=test_path,
                top_k=1,
                neighbors_k=5,
                relevant_threshold=4.0,
                output_path=results_path,
            )
            self.assertEqual(metrics["usersEvaluated"], 1)
            self.assertAlmostEqual(metrics["precisionAtK"], 1.0, places=5)
            self.assertAlmostEqual(metrics["recallAtK"], 1.0, places=5)
            self.assertTrue(results_path.exists())


if __name__ == "__main__":
    unittest.main()
