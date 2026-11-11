"""Ranking evaluation (Precision@K and Recall@K) against held-out ratings."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.config import TEST_CSV, EVALUATION_CSV, ensure_directories
from src.graph_store import Neo4jStore
from src.recommender import recommend_for_user


def user_precision_recall(recommended_ids: list[int], relevant_ids: set[int], k: int) -> tuple[float, float, int]:
    """Precision@K uses K as denominator; recall is undefined for an empty relevant set."""
    if k < 1:
        raise ValueError("k deve ser >= 1.")
    hits = len(set(recommended_ids[:k]) & relevant_ids)
    precision = hits / k
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall, hits


def evaluate_model(
    store: Neo4jStore,
    test_path: Path = TEST_CSV,
    top_k: int = 10,
    neighbors_k: int = 20,
    relevant_threshold: float = 4.0,
    output_path: Path = EVALUATION_CSV,
) -> dict[str, float | int]:
    if not test_path.exists():
        raise FileNotFoundError(f"Teste não encontrado: {test_path}. Execute 'python main.py prepare'.")
    if top_k < 1 or neighbors_k < 1:
        raise ValueError("top_k e neighbors_k precisam ser >= 1.")

    relevant_by_user: dict[int, set[int]] = defaultdict(set)
    with test_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if float(row["rating"]) >= relevant_threshold:
                relevant_by_user[int(row["userId"])].add(int(row["movieId"]))

    if not relevant_by_user:
        raise ValueError(
            f"O conjunto de teste não contém notas >= {relevant_threshold}. "
            "Confira relevant_threshold ou a divisão dos dados."
        )

    ensure_directories()
    results: list[dict[str, int | float]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    users = sorted(relevant_by_user)
    for index, user_id in enumerate(users, start=1):
        recommendations = recommend_for_user(store, user_id, k=neighbors_k, limit=top_k)
        recommended_ids = [item["movieId"] for item in recommendations]
        precision, recall, hits = user_precision_recall(
            recommended_ids, relevant_by_user[user_id], top_k
        )
        precisions.append(precision)
        recalls.append(recall)
        results.append({
            "userId": user_id,
            "relevantInTest": len(relevant_by_user[user_id]),
            "recommendationsReturned": len(recommended_ids),
            "hits": hits,
            "precisionAtK": precision,
            "recallAtK": recall,
        })
        if index == len(users) or index % 50 == 0:
            print(f"Avaliação: {index}/{len(users)} usuários")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["userId", "relevantInTest", "recommendationsReturned", "hits", "precisionAtK", "recallAtK"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary: dict[str, float | int] = {
        "usersEvaluated": len(results),
        "topK": top_k,
        "neighborsK": neighbors_k,
        "relevantThreshold": relevant_threshold,
        "precisionAtK": mean(precisions) if precisions else 0.0,
        "recallAtK": mean(recalls) if recalls else 0.0,
    }
    print("Métricas macro (média por usuário com ao menos um item relevante no teste):")
    print(f"  Precision@{top_k}: {summary['precisionAtK']:.4f}")
    print(f"  Recall@{top_k}:    {summary['recallAtK']:.4f}")
    print(f"Resultados por usuário: {output_path}")
    return summary
