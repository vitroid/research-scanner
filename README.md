# research-scanner

Semantic Scholar から特定分野の論文を取得し、BERTopic で abstract をトピッククラスタリングして、孤立論文・トピック間ギャップを検出するパイプライン。

## 何ができるか

- Semantic Scholar Graph API の bulk 検索で論文メタデータ + abstract を取得
- BERTopic (Sentence-Transformers + UMAP + HDBSCAN) で abstract を自動分類
- 孤立論文(BERTopic の outlier ∪ k-NN 距離が上位 5%)を抽出
- Plotly の 2D / 3D 散布図 + HTML レポートで可視化(マーカーを指すと右側パネルに概要が固定表示)

## セットアップ

Python 3.10+ 推奨。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Semantic Scholar API キー(任意・なくても動くがレート制限が緩くなる)
を持っている場合:

```bash
export SEMANTIC_SCHOLAR_API_KEY=...
```

## 実行例

`PLAN.md` の例(Bernal-Fowler ice rules)を 2010 年以降で最大 500 件:

```bash
python scripts/run_pipeline.py \
  --query "Bernal-Fowler ice rules" \
  --year-from 2010 \
  --max-papers 500
```

別分野に切り替えるときは `--query` と `--year-from` を変えるだけ。
キャッシュを無視して再取得するときは `--refresh`。

## 出力

`outputs/` 配下に以下が生成される:

| ファイル | 内容 |
|---|---|
| `report.html` | トピック一覧 + 孤立論文を 1 ページにまとめた HTML |
| `landscape.html` | UMAP 散布図(`--landscape-dim` で 2D / 3D 切替)|
| `topics.csv` | BERTopic の各トピックと代表語 |
| `papers_with_topics.csv` | 全論文 + トピック ID + isolation_score |
| `isolated_papers.csv` | 孤立フラグが立った論文だけ |

## 主要パラメータ

| オプション | デフォルト | 説明 |
|---|---|---|
| `--query` | `Bernal-Fowler ice rules` | Semantic Scholar 検索クエリ |
| `--year-from` / `--year-to` | `2010` / なし | 出版年フィルタ |
| `--max-papers` | `500` | 取得上限 |
| `--embedding-model` | `all-MiniLM-L6-v2` | 埋め込みモデル。科学論文特化なら `allenai/specter2_base` |
| `--min-cluster-size` | `10` | HDBSCAN の最小クラスタサイズ。論文数が多い時は大きく(目安: `sqrt(n)` ~ `n/30`) |
| `--cluster-selection-method` | `leaf` | `leaf`=細かい多数クラスタ(推奨)/ `eom`=少数の大きいクラスタ |
| `--isolation-percentile` | `95` | 孤立判定に使う k-NN 距離の上位パーセンタイル |
| `--landscape-dim` | `2` | landscape の次元(`2`=散布図 / `3`=回転可能な 3D 散布図)|

### パラメータチューニング

`scripts/tune.py` で `min_cluster_size` × `cluster_selection_method` をスイープして
outlier 率・トピック数・silhouette score を一覧できる。
キャッシュ済み `data/papers.parquet` を再利用するので試行が速い。

```bash
python scripts/tune.py --sweep 6 10 15 20 25 --methods leaf eom
```

実測例 (`spin ice`, 527 件):

| method | mcs | n_topics | outlier_pct | max_size | silhouette |
|---|---|---|---|---|---|
| eom  |  8 | 11 | 21.4 | 108 | -    |
| eom  | 15 |  5 |  9.7 | 382 | 0.31 ← 巨大クラスタ |
| leaf | 10 | 12 | 26.6 | 107 | 0.11 ← デフォルト |
| leaf | 15 |  9 | 41.9 |  93 | 0.14 |

`eom` は `min_cluster_size>=10` で 1 個の巨大クラスタが他を吸収しがち
(silhouette は高いが解釈性は低い)。`leaf` の方が研究分野の細分化に向く。

## モジュール構成

```
src/
  fetch.py          # Semantic Scholar クライアント
  topic_model.py    # BERTopic パイプライン
  gap_analysis.py   # 孤立論文 / トピック間ギャップ
  visualize.py      # Plotly + HTML レポート
scripts/
  run_pipeline.py   # エンドツーエンド
notebooks/
  research_scan.ipynb # 同じパイプラインをセル分割した notebook
```

各モジュールは独立しているので、Jupyter から個別に呼び出して
パラメータを試行錯誤するのも容易。

## Notebook / Colab

`notebooks/research_scan.ipynb` は `scripts/run_pipeline.py` と同じ処理を fetch / topic /
isolation / visualize のセルに分割したもの。

```bash
.venv/bin/python -m jupyter lab notebooks/research_scan.ipynb
```

Google Colab で開くと、notebook 先頭の "Colab セットアップ" セルが GitHub からこのリポジトリを
clone し、`requirements.txt` を `pip install` する。
`REPO_URL` を自分の fork に差し替えれば、Colab 上だけで一通り完結する。
