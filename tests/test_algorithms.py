import csv
import tempfile
import unittest
from pathlib import Path

from src.similarity import pearson_from_sums, compute_similarity_csv
from src.recommender import predict_rating
from src.evaluation import user_precision_recall
from src.data_pipeline import prepare_split


class PearsonTests(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        # x=[1,2,3], y=[2,4,6]
        n = 3
        sx, sy = 6, 12
        sx2, sy2, sxy = 14, 56, 28
        self.assertAlmostEqual(pearson_from_sums(n, sx, sy, sx2, sy2, sxy), 1.0)

    def test_perfect_negative_correlation(self):
        # x=[1,2,3], y=[3,2,1]
        result = pearson_from_sums(3, 6, 6, 14, 14, 10)
        self.assertAlmostEqual(result, -1.0)

    def test_constant_vector_is_undefined(self):
        self.assertIsNone(pearson_from_sums(3, 6, 9, 12, 27, 18))


class PredictionTests(unittest.TestCase):
    def test_weighted_prediction(self):
        result = predict_rating([(0.9, 5.0), (0.4, 3.0)])
        self.assertAlmostEqual(result, 4.3846153846)

    def test_prediction_is_clipped_to_rating_range(self):
        self.assertEqual(predict_rating([(1.0, 10.0)]), 5.0)
        self.assertEqual(predict_rating([(1.0, -2.0)]), 0.5)
        self.assertIsNone(predict_rating([(0.0, 4.0)]))


class EvaluationTests(unittest.TestCase):
    def test_precision_and_recall_at_k(self):
        precision, recall, hits = user_precision_recall([1, 2, 3, 4], {2, 4, 8, 9}, 4)
        self.assertEqual(hits, 2)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)

    def test_empty_recommendations_return_zero(self):
        precision, recall, hits = user_precision_recall([], {1, 2}, 10)
        self.assertEqual((precision, recall, hits), (0.0, 0.0, 0))


class DataPipelineTests(unittest.TestCase):
    def test_split_is_reproducible_and_80_20(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "ratings.csv"
            rows = [{"userId": i % 5 + 1, "movieId": i + 1, "rating": 4.0, "timestamp": 1000 + i} for i in range(100)]
            with raw.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["userId", "movieId", "rating", "timestamp"])
                writer.writeheader()
                writer.writerows(rows)
            # Exercise split mechanics without relying on the real dataset by temporarily patching module paths.
            import src.data_pipeline as pipeline
            old_ratings, old_movies = pipeline.RATINGS_CSV, pipeline.MOVIES_CSV
            old_train, old_test, old_raw, old_processed = pipeline.TRAIN_CSV, pipeline.TEST_CSV, pipeline.RAW_DIR, pipeline.PROCESSED_DIR
            try:
                pipeline.RATINGS_CSV = raw
                pipeline.MOVIES_CSV = root / "movies.csv"
                with pipeline.MOVIES_CSV.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["movieId", "title", "genres"])
                    writer.writeheader()
                    writer.writerow({"movieId": 1, "title": "Example", "genres": "Comedy"})
                pipeline.TRAIN_CSV = root / "processed" / "train.csv"
                pipeline.TEST_CSV = root / "processed" / "test.csv"
                pipeline.RAW_DIR = root
                pipeline.PROCESSED_DIR = root / "processed"
                pipeline.prepare_split(test_size=0.2, seed=7)
                with pipeline.TRAIN_CSV.open(encoding="utf-8") as f:
                    train1 = list(csv.DictReader(f))
                with pipeline.TEST_CSV.open(encoding="utf-8") as f:
                    test1 = list(csv.DictReader(f))
                pipeline.prepare_split(test_size=0.2, seed=7)
                with pipeline.TRAIN_CSV.open(encoding="utf-8") as f:
                    train2 = list(csv.DictReader(f))
                self.assertEqual(len(train1), 80)
                self.assertEqual(len(test1), 20)
                self.assertEqual(train1, train2)
                self.assertEqual({r["movieId"] for r in train1} & {r["movieId"] for r in test1}, set())
            finally:
                pipeline.RATINGS_CSV, pipeline.MOVIES_CSV = old_ratings, old_movies
                pipeline.TRAIN_CSV, pipeline.TEST_CSV = old_train, old_test
                pipeline.RAW_DIR, pipeline.PROCESSED_DIR = old_raw, old_processed

    def test_similarity_csv_with_tiny_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            rows = [
                {"userId": 1, "movieId": 1, "rating": 1, "timestamp": 0},
                {"userId": 1, "movieId": 2, "rating": 2, "timestamp": 0},
                {"userId": 2, "movieId": 1, "rating": 3, "timestamp": 0},
                {"userId": 2, "movieId": 2, "rating": 4, "timestamp": 0},
                {"userId": 3, "movieId": 1, "rating": 4, "timestamp": 0},
                {"userId": 3, "movieId": 2, "rating": 5, "timestamp": 0},
            ]
            with train.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["userId", "movieId", "rating", "timestamp"])
                writer.writeheader()
                writer.writerows(rows)
            output = root / "similarities.csv"
            path, count = compute_similarity_csv(train, output, min_common_users=2)
            with path.open(encoding="utf-8") as f:
                results = list(csv.DictReader(f))
            self.assertEqual(count, 1)
            self.assertEqual((int(results[0]["movieId1"]), int(results[0]["movieId2"])), (1, 2))
            self.assertAlmostEqual(float(results[0]["correlation"]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
