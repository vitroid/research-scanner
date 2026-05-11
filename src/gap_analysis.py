"""Identify isolated papers and large gaps between topic clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


@dataclass
class GapConfig:
    isolation_percentile: float = 95.0
    knn: int = 5


def isolated_papers(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cfg: GapConfig | None = None,
) -> pd.DataFrame:
    """Return papers ranked by isolation (mean cosine distance to k nearest neighbors).

    Combines two signals:
      * BERTopic outliers (`topic == -1`)
      * Top percentile of mean k-NN distance in embedding space
    """
    cfg = cfg or GapConfig()
    if len(df) == 0:
        return df.assign(isolation_score=[], is_isolated=[])

    k = min(cfg.knn + 1, len(embeddings))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    mean_dist = distances[:, 1:].mean(axis=1) if k > 1 else distances[:, 0]

    threshold = np.percentile(mean_dist, cfg.isolation_percentile)
    is_outlier = df["topic"].to_numpy() == -1 if "topic" in df.columns else np.zeros(
        len(df), dtype=bool
    )
    is_isolated = (mean_dist >= threshold) | is_outlier

    out = df.copy()
    out["isolation_score"] = mean_dist
    out["is_isolated"] = is_isolated
    out = out.sort_values("isolation_score", ascending=False).reset_index(drop=True)
    logger.info(
        "Flagged %d / %d papers as isolated (>= p%.0f or BERTopic outlier).",
        int(out["is_isolated"].sum()),
        len(out),
        cfg.isolation_percentile,
    )
    return out


def topic_centroid_distances(
    embeddings: np.ndarray,
    topics: list[int] | np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute pairwise cosine distance between topic centroids.

    Returns the distance matrix plus a long-form DataFrame of the largest gaps
    (which can be read as "categories that sit far apart in embedding space").
    """
    topics_arr = np.asarray(topics)
    unique = sorted(int(t) for t in set(topics_arr) if t != -1)
    if not unique:
        return np.empty((0, 0)), pd.DataFrame(columns=["topic_a", "topic_b", "distance"])

    centroids = np.vstack(
        [embeddings[topics_arr == t].mean(axis=0) for t in unique]
    )
    dist = cosine_distances(centroids)

    rows: list[dict] = []
    for i, a in enumerate(unique):
        for j, b in enumerate(unique):
            if j <= i:
                continue
            rows.append({"topic_a": a, "topic_b": b, "distance": float(dist[i, j])})
    pairs = (
        pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)
    )
    return dist, pairs


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
