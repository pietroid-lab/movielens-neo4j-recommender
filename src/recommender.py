"""Item-based KNN prediction and top-N recommendations through Neo4j."""
from __future__ import annotations

from typing import Any

from src.graph_store import Neo4jStore

RECOMMEND_QUERY = """
MATCH (u:User {userId: $userId})-[r:RATED]->(neighbor:Movie)-[s:SIMILAR]-(candidate:Movie)
WHERE s.correlation <> 0.0
  AND NOT EXISTS { MATCH (u)-[:RATED]->(candidate) }
WITH candidate, r.rating AS neighborRating, s.correlation AS correlation
ORDER BY candidate.movieId, abs(correlation) DESC
WITH candidate,
     collect({rating: neighborRating, correlation: correlation})[0..$k] AS neighbors
WITH candidate, neighbors,
     reduce(num = 0.0, n IN neighbors | num + n.correlation * n.rating) AS numerator,
     reduce(den = 0.0, n IN neighbors | den + abs(n.correlation)) AS denominator
WHERE denominator > 0.0
WITH candidate,
     CASE
       WHEN numerator / denominator < 0.5 THEN 0.5
       WHEN numerator / denominator > 5.0 THEN 5.0
       ELSE numerator / denominator
     END AS predictedRating,
     size(neighbors) AS neighborCount
RETURN candidate.movieId AS movieId,
       candidate.title AS title,
       predictedRating,
       neighborCount
ORDER BY predictedRating DESC, title ASC
LIMIT $limit
"""


def predict_rating(neighbors: list[tuple[float, float]]) -> float | None:
    """Pure-Python weighted Pearson estimate; each tuple is (correlation, rating)."""
    numerator = sum(correlation * rating for correlation, rating in neighbors)
    denominator = sum(abs(correlation) for correlation, _ in neighbors)
    if denominator <= 0.0:
        return None
    return min(5.0, max(0.5, numerator / denominator))


def recommend_for_user(store: Neo4jStore, user_id: int, k: int = 20, limit: int = 10) -> list[dict[str, Any]]:
    if k < 1:
        raise ValueError("k deve ser >= 1.")
    if limit < 1:
        raise ValueError("limit deve ser >= 1.")
    records, _, _ = store.execute(RECOMMEND_QUERY, userId=user_id, k=k, limit=limit)
    return [
        {
            "movieId": int(record["movieId"]),
            "title": record["title"],
            "predictedRating": float(record["predictedRating"]),
            "neighborCount": int(record["neighborCount"]),
        }
        for record in records
    ]
