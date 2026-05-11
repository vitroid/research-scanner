"""Semantic Scholar Graph API client for bulk paper retrieval."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

DEFAULT_FIELDS = (
    "paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "authors.name",
    "citationCount",
    "referenceCount",
    "externalIds",
    "publicationDate",
)


@dataclass
class FetchConfig:
    query: str
    year_from: int | None = None
    year_to: int | None = None
    max_papers: int = 1000
    fields: tuple[str, ...] = DEFAULT_FIELDS
    sleep_seconds: float = 1.1
    api_key: str | None = None


def _build_params(cfg: FetchConfig, token: str | None) -> dict[str, str]:
    params: dict[str, str] = {
        "query": cfg.query,
        "fields": ",".join(cfg.fields),
    }
    if cfg.year_from is not None or cfg.year_to is not None:
        lo = cfg.year_from if cfg.year_from is not None else ""
        hi = cfg.year_to if cfg.year_to is not None else ""
        params["year"] = f"{lo}-{hi}"
    if token:
        params["token"] = token
    return params


def _flatten_record(raw: dict) -> dict:
    authors = raw.get("authors") or []
    ext = raw.get("externalIds") or {}
    return {
        "paper_id": raw.get("paperId"),
        "title": raw.get("title"),
        "abstract": raw.get("abstract"),
        "year": raw.get("year"),
        "publication_date": raw.get("publicationDate"),
        "venue": raw.get("venue"),
        "authors": "; ".join(a.get("name", "") for a in authors),
        "n_authors": len(authors),
        "citation_count": raw.get("citationCount"),
        "reference_count": raw.get("referenceCount"),
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
    }


def fetch_papers(cfg: FetchConfig) -> pd.DataFrame:
    """Fetch papers via the Semantic Scholar bulk search endpoint.

    The bulk endpoint paginates with a `token` cursor and returns up to 1000
    records per call.
    """
    headers: dict[str, str] = {}
    api_key = cfg.api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    records: list[dict] = []
    token: str | None = None
    pbar = tqdm(total=cfg.max_papers, desc="Fetching papers", unit="paper")

    while len(records) < cfg.max_papers:
        params = _build_params(cfg, token)
        try:
            resp = requests.get(
                BULK_SEARCH_URL, params=params, headers=headers, timeout=60
            )
        except requests.RequestException as exc:
            logger.warning("Request failed: %s. Retrying after backoff.", exc)
            time.sleep(5)
            continue

        if resp.status_code == 429:
            logger.warning("Rate limited; sleeping 10s.")
            time.sleep(10)
            continue
        resp.raise_for_status()

        payload = resp.json()
        batch: Iterable[dict] = payload.get("data") or []
        added = 0
        for raw in batch:
            records.append(_flatten_record(raw))
            added += 1
            if len(records) >= cfg.max_papers:
                break
        pbar.update(added)

        token = payload.get("token")
        if not token or added == 0:
            break
        time.sleep(cfg.sleep_seconds)

    pbar.close()
    df = pd.DataFrame.from_records(records)
    logger.info("Fetched %d papers (raw).", len(df))
    return df


def filter_with_abstract(df: pd.DataFrame, min_chars: int = 100) -> pd.DataFrame:
    """Keep only rows whose abstract is informative enough for topic modeling."""
    if df.empty:
        return df
    mask = df["abstract"].notna() & (df["abstract"].str.len() >= min_chars)
    filtered = df.loc[mask].drop_duplicates(subset="paper_id").reset_index(drop=True)
    logger.info(
        "Filtered to %d papers with abstracts (>= %d chars).", len(filtered), min_chars
    )
    return filtered


def exclude_keywords(
    df: pd.DataFrame,
    keywords: list[str] | tuple[str, ...],
    fields: tuple[str, ...] = ("title", "abstract"),
) -> pd.DataFrame:
    """Drop rows whose `fields` contain any of `keywords` (case-insensitive, whole-word).

    Whole-word matching is preferred over substring so that, e.g., excluding
    "spin" does not also drop "spinach" or "spinodal".
    """
    if df.empty or not keywords:
        return df
    import re

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
        flags=re.IGNORECASE,
    )

    def _hit(row: pd.Series) -> bool:
        for f in fields:
            val = row.get(f)
            if isinstance(val, str) and pattern.search(val):
                return True
        return False

    mask = df.apply(_hit, axis=1)
    kept = df.loc[~mask].reset_index(drop=True)
    logger.info(
        "Excluded %d / %d papers matching keywords %s (in %s).",
        int(mask.sum()),
        len(df),
        list(keywords),
        list(fields),
    )
    return kept


def save_papers(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d papers to %s.", len(df), path)
    return path


def load_papers(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
