"""End-to-end research scan: fetch -> topic model -> isolation -> report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fetch import (  # noqa: E402
    FetchConfig,
    exclude_keywords,
    fetch_papers,
    filter_with_abstract,
    load_papers,
    save_papers,
)
from src.gap_analysis import (  # noqa: E402
    GapConfig,
    isolated_papers,
    save_table,
)
from src.topic_model import (  # noqa: E402
    TopicModelConfig,
    attach_topic_columns,
    fit_topic_model,
    save_topic_info,
)
from src.visualize import build_report, landscape_figure, save_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--query",
        default="Bernal-Fowler ice rules",
        help="Search query passed to Semantic Scholar.",
    )
    p.add_argument("--year-from", type=int, default=2010)
    p.add_argument("--year-to", type=int, default=None)
    p.add_argument("--max-papers", type=int, default=500)
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name. Try 'allenai/specter2_base' for science.",
    )
    p.add_argument("--min-cluster-size", type=int, default=10)
    p.add_argument(
        "--cluster-selection-method",
        choices=["leaf", "eom"],
        default="leaf",
        help="HDBSCAN cluster selection. 'leaf' = many fine clusters (recommended), 'eom' = fewer large clusters.",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "data" / "papers.parquet",
        help="Where to cache the fetched paper dataframe.",
    )
    p.add_argument(
        "--outputs",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory to write report artifacts.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cache and refetch from Semantic Scholar.",
    )
    p.add_argument("--isolation-percentile", type=float, default=95.0)
    p.add_argument(
        "--exclude-keywords",
        nargs="*",
        default=[],
        help="Drop papers whose title or abstract contains any of these whole-words (case-insensitive).",
    )
    p.add_argument(
        "--landscape-dim",
        type=int,
        choices=[2, 3],
        default=2,
        help="Dimensionality of the landscape plot (2 = scatter, 3 = rotatable 3D scatter).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("pipeline")

    args.outputs.mkdir(parents=True, exist_ok=True)

    if args.cache.exists() and not args.refresh:
        log.info("Loading cached papers from %s.", args.cache)
        papers = load_papers(args.cache)
    else:
        cfg = FetchConfig(
            query=args.query,
            year_from=args.year_from,
            year_to=args.year_to,
            max_papers=args.max_papers,
        )
        papers = fetch_papers(cfg)
        save_papers(papers, args.cache)

    papers = filter_with_abstract(papers)
    if args.exclude_keywords:
        papers = exclude_keywords(papers, args.exclude_keywords)
    if len(papers) < 10:
        log.error(
            "Too few papers with abstracts (%d). Broaden the query or year range.",
            len(papers),
        )
        return 1

    tm_cfg = TopicModelConfig(
        embedding_model=args.embedding_model,
        min_cluster_size=args.min_cluster_size,
        cluster_selection_method=args.cluster_selection_method,
        coords_dim=args.landscape_dim,
    )
    result = fit_topic_model(papers["abstract"].tolist(), cfg=tm_cfg)
    papers = attach_topic_columns(papers, result)

    gap_cfg = GapConfig(isolation_percentile=args.isolation_percentile)
    isolated = isolated_papers(papers, result.embeddings, gap_cfg)

    fig = landscape_figure(isolated)
    figure_path = save_figure(fig, args.outputs / "landscape.html")
    save_topic_info(result, args.outputs / "topics.csv")
    save_table(isolated, args.outputs / "papers_with_topics.csv")
    save_table(
        isolated.loc[isolated["is_isolated"]], args.outputs / "isolated_papers.csv"
    )

    report_path = build_report(
        query=args.query,
        n_papers=len(papers),
        figure_path=figure_path,
        topic_info=result.topic_info,
        isolated_df=isolated.loc[isolated["is_isolated"]],
        output_path=args.outputs / "report.html",
    )
    log.info("Report written to %s.", report_path)

    print("\n=== Summary ===")
    print(f"Query           : {args.query}")
    print(f"Papers analyzed : {len(papers)}")
    print(f"Topics found    : {(result.topic_info['Topic'] >= 0).sum()}")
    print(
        f"Isolated papers : {int(isolated['is_isolated'].sum())} (top {gap_cfg.isolation_percentile:.0f}p + BERTopic outliers)"
    )
    print(f"Report          : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
