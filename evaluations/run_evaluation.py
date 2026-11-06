
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, EVALUATION_DIR, GRAPH_DIR
from src.data_loader import load_news_data
from src.preprocessing import preprocess_dataframe, clean_text
from src.tfidf_search import TFIDFRetriever
from src.bm25_search import BM25Retriever
from src.semantic_search import SemanticRetriever
from src.hybrid_search import reciprocal_rank_fusion
from src.evaluation import evaluate_ranking


QUERIES = [
    ("football team match players", "sport"),
    ("government election parliament policy", "politics"),
    ("computer technology software internet", "tech"),
    ("business economy market company", "business"),
    ("film music celebrity television", "entertainment")
]


def main():

    EVALUATION_DIR.mkdir(exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = [
        p for p in DATA_DIR.glob("*.csv")
        if p.name != "processed_data.csv"
    ]

    if not csv_files:
        raise FileNotFoundError(
            "Put your BBC CSV file inside the data folder."
        )

    df = load_news_data(csv_files[0])
    df = preprocess_dataframe(df)

    df.to_csv(
        DATA_DIR / "processed_data.csv",
        index=False
    )

    documents = df["clean_text"].tolist()

    print("\nBuilding retrieval indexes...")

    tfidf = TFIDFRetriever(documents)
    bm25 = BM25Retriever(documents)

    semantic = SemanticRetriever(
        df["document"].tolist()
    )

    results = []

    for query, category in QUERIES:

        matching = (
            df["category"].str.lower()
            .str.contains(category, regex=False)
        )

        relevant_ids = set(
            df.index[matching].tolist()
        )

        if not relevant_ids:
            print("Skipping unavailable category:", category)
            continue

        clean_query = clean_text(query)

        tfidf_results = tfidf.search(
            clean_query, len(df)
        )

        bm25_results = bm25.search(
            clean_query, len(df)
        )

        semantic_results = semantic.search(
            query, len(df)
        )

        hybrid_results = reciprocal_rank_fusion(
            [
                tfidf_results,
                bm25_results,
                semantic_results
            ],
            k=60,
            top_k=len(df)
        )

        model_results = {
            "TF-IDF": tfidf_results,
            "BM25": bm25_results,
            "Sentence Transformer": semantic_results,
            "Hybrid RRF": hybrid_results
        }

        for model_name, ranking in model_results.items():

            ranked_ids = [
                doc_id for doc_id, score in ranking
            ]

            metrics = evaluate_ranking(
                ranked_ids,
                relevant_ids,
                k=10
            )

            results.append({
                "Query": query,
                "Expected Category": category,
                "Model": model_name,
                **metrics
            })

            print(
                f"{model_name} evaluated for: {query}"
            )

    if not results:
        raise ValueError(
            "No evaluation queries matched the dataset categories. "
            "Check your category names and edit QUERIES."
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        EVALUATION_DIR / "results.csv",
        index=False
    )

    summary = results_df.groupby(
        "Model"
    ).mean(numeric_only=True)

    summary.to_csv(
        EVALUATION_DIR / "model_summary.csv"
    )

    print("\nMODEL PERFORMANCE COMPARISON")
    print(summary.round(4))

    # Metric comparison graphs
    for metric in summary.columns:

        plt.figure(figsize=(10, 6))

        summary[metric].plot(
            kind="bar",
            rot=20
        )

        plt.title(f"Model Comparison - {metric}")
        plt.xlabel("Retrieval Model")
        plt.ylabel(metric)
        plt.tight_layout()

        plt.savefig(
            GRAPH_DIR / f"{metric}.png",
            dpi=150
        )

        plt.close()

    # Precision-Recall comparison graph
    plt.figure(figsize=(9, 6))

    for model_name, group in results_df.groupby("Model"):

        plt.plot(
            group["Recall@10"],
            group["Precision@10"],
            marker="o",
            label=model_name
        )

    plt.title("Precision-Recall Comparison")
    plt.xlabel("Recall@10")
    plt.ylabel("Precision@10")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "precision_recall.png",
        dpi=150
    )

    plt.close()

    print("\nEvaluation completed successfully!")
    print("Results saved in evaluation/results.csv")
    print("Graphs saved in evaluation/graphs/")


if __name__ == "__main__":
    main()
