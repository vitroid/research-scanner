"""BERTopic-based topic modeling pipeline for paper abstracts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

logger = logging.getLogger(__name__)


@dataclass
class TopicModelConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_cluster_size: int = 10
    min_samples: int | None = None
    cluster_selection_method: str = "leaf"
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    random_state: int = 42
    nr_topics: str | int | None = None
    stop_words: str = "english"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    extra_stop_words: list[str] = field(default_factory=list)
    # Dimensionality of the final UMAP projection used for the landscape plot
    # (2 = scatter, 3 = rotatable Scatter3d). The clustering UMAP above is
    # independent and stays at `umap_n_components`.
    coords_dim: int = 2


@dataclass
class TopicModelResult:
    model: BERTopic
    embeddings: np.ndarray
    topics: list[int]
    probabilities: np.ndarray | None
    topic_info: pd.DataFrame
    coords: np.ndarray  # shape (n_docs, coords_dim) — 2D or 3D depending on config


def _build_vectorizer(cfg: TopicModelConfig) -> CountVectorizer:
    from sklearn.feature_extraction import text as sk_text

    base = list(sk_text.ENGLISH_STOP_WORDS) if cfg.stop_words == "english" else []
    stop_words = sorted(set(base) | set(cfg.extra_stop_words))
    return CountVectorizer(
        stop_words=stop_words,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
    )


def embed_documents(docs: list[str], cfg: TopicModelConfig) -> np.ndarray:
    logger.info("Embedding %d documents with %s.", len(docs), cfg.embedding_model)
    model = SentenceTransformer(cfg.embedding_model)
    return model.encode(docs, show_progress_bar=True, convert_to_numpy=True)


def fit_topic_model(
    docs: list[str],
    embeddings: np.ndarray | None = None,
    cfg: TopicModelConfig | None = None,
) -> TopicModelResult:
    cfg = cfg or TopicModelConfig()
    if embeddings is None:
        embeddings = embed_documents(docs, cfg)

    umap_model = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=cfg.umap_n_components,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric="euclidean",
        cluster_selection_method=cfg.cluster_selection_method,
        prediction_data=True,
    )
    vectorizer = _build_vectorizer(cfg)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=cfg.nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    topic_info = topic_model.get_topic_info()

    if cfg.coords_dim not in (2, 3):
        raise ValueError(f"coords_dim must be 2 or 3, got {cfg.coords_dim}")
    coords = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=cfg.coords_dim,
        min_dist=0.1,
        metric="cosine",
        random_state=cfg.random_state,
    ).fit_transform(embeddings)

    n_topics = (topic_info["Topic"] >= 0).sum()
    n_outliers = int((np.array(topics) == -1).sum())
    logger.info(
        "BERTopic produced %d topics (%d outliers / %d docs); landscape is %dD.",
        n_topics,
        n_outliers,
        len(docs),
        cfg.coords_dim,
    )

    return TopicModelResult(
        model=topic_model,
        embeddings=embeddings,
        topics=list(topics),
        probabilities=probs if isinstance(probs, np.ndarray) else None,
        topic_info=topic_info,
        coords=np.asarray(coords),
    )


def topic_label(topic_info: pd.DataFrame, topic_id: int, top_k: int = 4) -> str:
    """Return a short human-readable label for a topic from its top c-TF-IDF terms."""
    if topic_id == -1:
        return "Outlier"
    row = topic_info.loc[topic_info["Topic"] == topic_id]
    if row.empty:
        return f"Topic {topic_id}"
    name = row.iloc[0]["Name"]
    parts = [p for p in name.split("_") if not p.isdigit()][:top_k]
    return ", ".join(parts) if parts else f"Topic {topic_id}"


def attach_topic_columns(
    df: pd.DataFrame, result: TopicModelResult
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["topic"] = result.topics
    out["topic_label"] = [
        topic_label(result.topic_info, int(t)) for t in result.topics
    ]
    out["umap_x"] = result.coords[:, 0]
    out["umap_y"] = result.coords[:, 1]
    if result.coords.shape[1] >= 3:
        out["umap_z"] = result.coords[:, 2]
    return out


def save_topic_info(result: TopicModelResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.topic_info.to_csv(path, index=False)
    return path
