"""Sweep HDBSCAN min_cluster_size and report quality metrics.

Reuses cached embeddings so each trial is fast.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fetch import filter_with_abstract, load_papers  # noqa: E402
from src.topic_model import (  # noqa: E402
    TopicModelConfig,
    embed_documents,
    fit_topic_model,
)


def silhouette_clustered(
    embeddings: np.ndarray, topics: list[int]
) -> float | None:
    """Silhouette score on the non-outlier subset (cosine, sampled)."""
    from sklearn.metrics import silhouette_score

    topics_arr = np.asarray(topics)
    mask = topics_arr != -1
    if mask.sum() < 50 or len(set(topics_arr[mask])) < 2:
        return None
    rng = np.random.default_rng(0)
    idx = np.where(mask)[0]
    if len(idx) > 1000:
        idx = rng.choice(idx, size=1000, replace=False)
    return float(
        silhouette_score(embeddings[idx], topics_arr[idx], metric="cosine")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "papers.parquet")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=[6, 10, 15, 20, 25, 30],
        help="min_cluster_size values to try.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["eom", "leaf"],
        help="HDBSCAN cluster_selection_method values to compare.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    papers = filter_with_abstract(load_papers(args.cache))
    docs = papers["abstract"].tolist()
    print(f"Documents: {len(docs)}")

    cfg = TopicModelConfig(embedding_model=args.embedding_model)
    embeddings = embed_documents(docs, cfg)

    rows = []
    for method in args.methods:
        for mcs in args.sweep:
            tm_cfg = TopicModelConfig(
                embedding_model=args.embedding_model,
                min_cluster_size=mcs,
                cluster_selection_method=method,
            )
            result = fit_topic_model(docs, embeddings=embeddings, cfg=tm_cfg)
            topics = np.asarray(result.topics)
            n_topics = int((result.topic_info["Topic"] >= 0).sum())
            n_outliers = int((topics == -1).sum())
            outlier_pct = 100.0 * n_outliers / len(topics)
            non_out = result.topic_info.loc[
                result.topic_info["Topic"] >= 0, "Count"
            ]
            median_size = float(non_out.median()) if len(non_out) else 0.0
            min_size = int(non_out.min()) if len(non_out) else 0
            max_size = int(non_out.max()) if len(non_out) else 0
            sil = silhouette_clustered(embeddings, result.topics)

            rows.append(
                {
                    "method": method,
                    "min_cluster_size": mcs,
                    "n_topics": n_topics,
                    "outlier_pct": round(outlier_pct, 1),
                    "median_topic_size": median_size,
                    "min_topic_size": min_size,
                    "max_topic_size": max_size,
                    "silhouette_cosine": round(sil, 3) if sil is not None else None,
                }
            )

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
