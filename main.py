"""CLI for the MovieLens + Neo4j recommender project."""
from __future__ import annotations

import argparse
import json
import sys

from src.config import MOVIES_CSV, RATINGS_CSV, TRAIN_CSV, SIMILARITIES_CSV
from src.data_pipeline import download_dataset, prepare_split
from src.similarity import compute_similarity_csv
from src.graph_store import Neo4jStore
from src.recommender import recommend_for_user
from src.evaluation import evaluate_model


def run_all(args: argparse.Namespace) -> None:
    download_dataset()
    prepare_split(test_size=args.test_size, seed=args.seed)
    compute_similarity_csv(min_common_users=args.min_common)
    with Neo4jStore() as store:
        store.import_graph(reset_graph=args.reset_graph)
    print("Pipeline concluído. Agora você pode executar 'recommend' e 'evaluate'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sistema de recomendação MovieLens com grafo Neo4j e KNN item-item.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download_parser = commands.add_parser("download", help="Baixa MovieLens latest-small.")
    download_parser.add_argument("--force", action="store_true", help="Baixa novamente mesmo que os CSVs existam.")

    prepare_parser = commands.add_parser("prepare", help="Cria a separação treino/teste.")
    prepare_parser.add_argument("--test-size", type=float, default=0.2, help="Fração reservada para teste (padrão: 0.2).")
    prepare_parser.add_argument("--seed", type=int, default=42, help="Semente para divisão reproduzível.")

    similarity_parser = commands.add_parser("similarities", help="Calcula Pearson entre pares de filmes no treino.")
    similarity_parser.add_argument("--min-common", type=int, default=3, help="Mínimo de usuários em comum por par (padrão: 3).")

    import_parser = commands.add_parser("import", help="Importa filmes, ratings de treino e similaridades no Neo4j.")
    import_parser.add_argument("--reset-graph", action="store_true", help="APAGA todos os nós/relacionamentos do banco configurado antes da importação.")
    import_parser.add_argument("--batch-size", type=int, default=1000, help="Quantidade de linhas por lote.")

    query_parser = commands.add_parser("comedies", help="Conta comédias que um usuário avaliou >= uma nota.")
    query_parser.add_argument("--user-id", type=int, required=True)
    query_parser.add_argument("--min-rating", type=float, default=4.0)

    recommend_parser = commands.add_parser("recommend", help="Gera recomendações top-N para um usuário.")
    recommend_parser.add_argument("--user-id", type=int, required=True)
    recommend_parser.add_argument("--k", type=int, default=20, help="Vizinhos usados por candidato.")
    recommend_parser.add_argument("--limit", type=int, default=10, help="Número de recomendações retornadas.")

    evaluate_parser = commands.add_parser("evaluate", help="Calcula Precision@K e Recall@K com o teste reservado.")
    evaluate_parser.add_argument("--k", type=int, default=20, help="Vizinhos usados por candidato.")
    evaluate_parser.add_argument("--top-n", type=int, default=10, help="Tamanho do ranking de recomendações.")
    evaluate_parser.add_argument("--threshold", type=float, default=4.0, help="Nota mínima para ser relevante.")

    all_parser = commands.add_parser("all", help="Executa download, split, Pearson e importação no Neo4j.")
    all_parser.add_argument("--test-size", type=float, default=0.2)
    all_parser.add_argument("--seed", type=int, default=42)
    all_parser.add_argument("--min-common", type=int, default=3)
    all_parser.add_argument("--reset-graph", action="store_true", help="APAGA o grafo existente antes de importar.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "download":
            download_dataset(force=args.force)
        elif args.command == "prepare":
            if not MOVIES_CSV.exists() or not RATINGS_CSV.exists():
                download_dataset()
            prepare_split(test_size=args.test_size, seed=args.seed)
        elif args.command == "similarities":
            compute_similarity_csv(min_common_users=args.min_common)
        elif args.command == "import":
            with Neo4jStore() as store:
                store.import_graph(reset_graph=args.reset_graph, batch_size=args.batch_size)
        elif args.command == "comedies":
            with Neo4jStore() as store:
                count = store.count_comedies(args.user_id, args.min_rating)
            print(f"Usuário {args.user_id}: {count} comédias com nota >= {args.min_rating:g}")
        elif args.command == "recommend":
            with Neo4jStore() as store:
                rows = recommend_for_user(store, args.user_id, k=args.k, limit=args.limit)
            if not rows:
                print("Nenhuma recomendação foi encontrada para esse usuário com as similaridades atuais.")
            else:
                for position, row in enumerate(rows, start=1):
                    print(f"{position:2}. {row['title']} (movieId={row['movieId']}) — previsão={row['predictedRating']:.2f}; vizinhos={row['neighborCount']}")
        elif args.command == "evaluate":
            with Neo4jStore() as store:
                summary = evaluate_model(
                    store,
                    top_k=args.top_n,
                    neighbors_k=args.k,
                    relevant_threshold=args.threshold,
                )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == "all":
            run_all(args)
        else:
            parser.error("Comando desconhecido.")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERRO inesperado: {exc}", file=sys.stderr)
        print("Confira se o Neo4j está ativo e se as variáveis do arquivo .env estão corretas.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
