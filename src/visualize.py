"""Plotly-based visualization of the topic landscape."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _truncate(text: str | None, limit: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Symbols chosen for maximum visual distinctness when paired with colors.
# Outlier is forced to a muted gray "x" so real topics get the bright slots.
_SYMBOL_SEQUENCE_2D: tuple[str, ...] = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "star",
    "pentagon",
    "hexagon",
    "cross",
    "hourglass",
    "bowtie",
    "star-square",
    "star-diamond",
    "diamond-tall",
    "hexagram",
    "triangle-left",
    "triangle-right",
    "octagon",
)

# Plotly's Scatter3d supports only a small symbol set. Color does most of
# the disambiguation; symbol just adds a second channel.
_SYMBOL_SEQUENCE_3D: tuple[str, ...] = (
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "circle-open",
    "square-open",
    "diamond-open",
)


def _topic_styles(labels: list[str], is_3d: bool = False) -> dict[str, dict[str, str]]:
    """Assign (color, symbol) pairs to each non-outlier topic deterministically."""
    palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
    )
    symbols = _SYMBOL_SEQUENCE_3D if is_3d else _SYMBOL_SEQUENCE_2D
    outlier_symbol = "x" if is_3d else "x-thin"
    styles: dict[str, dict[str, str]] = {}
    real_idx = 0
    for label in labels:
        if label == "Outlier":
            styles[label] = {"color": "#9e9e9e", "symbol": outlier_symbol}
            continue
        styles[label] = {
            "color": palette[real_idx % len(palette)],
            "symbol": symbols[real_idx % len(symbols)],
        }
        real_idx += 1
    return styles


def landscape_figure(df: pd.DataFrame) -> go.Figure:
    """UMAP scatter colored AND shaped by topic, with isolated papers highlighted.

    Produces a 2D `Scatter` figure by default. If the dataframe has a
    ``umap_z`` column, a rotatable 3D `Scatter3d` is produced instead.
    Hover details are routed to the sidebar info panel via `plotly_hover`
    in `save_figure`; both 2D and 3D scatters fire that event.
    """
    plot_df = df.copy()
    plot_df["title_short"] = plot_df["title"].apply(lambda t: _truncate(t, 100))
    plot_df["abstract_short"] = plot_df["abstract"].apply(lambda t: _truncate(t, 240))
    plot_df["topic_label"] = plot_df["topic_label"].fillna("Outlier")

    is_3d = "umap_z" in plot_df.columns

    counts = plot_df["topic_label"].value_counts()
    ordered_labels = [lbl for lbl in counts.index if lbl != "Outlier"]
    if "Outlier" in counts.index:
        ordered_labels.append("Outlier")
    styles = _topic_styles(ordered_labels, is_3d=is_3d)

    cd_cols = ["title_short", "year", "citation_count", "abstract_short", "doi"]

    fig = go.Figure()
    for label in ordered_labels:
        sub = plot_df.loc[plot_df["topic_label"] == label]
        is_outlier = label == "Outlier"
        customdata = sub[cd_cols].fillna("").to_numpy()
        if is_3d:
            marker = dict(
                size=4 if not is_outlier else 3,
                color=styles[label]["color"],
                symbol=styles[label]["symbol"],
                opacity=0.85 if not is_outlier else 0.5,
                line=dict(width=0.5, color="rgba(40,40,40,0.6)"),
            )
            fig.add_trace(
                go.Scatter3d(
                    x=sub["umap_x"],
                    y=sub["umap_y"],
                    z=sub["umap_z"],
                    mode="markers",
                    name=f"{label}  (n={len(sub)})",
                    marker=marker,
                    customdata=customdata,
                    hoverinfo="none",
                    hovertemplate=None,
                )
            )
        else:
            marker = dict(
                size=8 if not is_outlier else 6,
                color=styles[label]["color"],
                symbol=styles[label]["symbol"],
                opacity=0.85 if not is_outlier else 0.5,
                line=dict(width=0.5, color="rgba(40,40,40,0.6)"),
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["umap_x"],
                    y=sub["umap_y"],
                    mode="markers",
                    name=f"{label}  (n={len(sub)})",
                    marker=marker,
                    customdata=customdata,
                    # Cursor-following tooltip is suppressed; the surrounding HTML
                    # listens to plotly_hover and renders details in a fixed panel.
                    hoverinfo="none",
                    hovertemplate=None,
                )
            )

    if "is_isolated" in plot_df.columns:
        iso = plot_df.loc[plot_df["is_isolated"]]
        if not iso.empty:
            iso_customdata = iso[cd_cols].fillna("").to_numpy()
            # Two overlapping traces give an unmistakable "target" look:
            # a bright outer ring + a centered X. In 3D the ring can be hidden
            # behind opaque topic markers via depth-buffer, but the X stays
            # visible from any angle, so at least one channel always reads.
            iso_color = "#e11d48"  # crimson — high contrast vs every palette color
            # NOTE on plotly marker colors:
            # "*-open" symbols (e.g. "circle-open") take their outline color
            # from `marker.color` and *ignore* `marker.line.color`. To get a
            # hollow ring whose outline color is controlled independently
            # from the fill, use the filled symbol with a transparent
            # `marker.color` and let `marker.line.color` paint the ring.
            transparent = "rgba(0,0,0,0)"
            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=iso["umap_x"],
                        y=iso["umap_y"],
                        z=iso["umap_z"],
                        mode="markers",
                        name=f"Isolated  (n={len(iso)})",
                        marker=dict(
                            size=14,
                            symbol="circle",
                            color=transparent,
                            line=dict(width=3, color=iso_color),
                        ),
                        customdata=iso_customdata,
                        hoverinfo="none",
                        hovertemplate=None,
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=iso["umap_x"],
                        y=iso["umap_y"],
                        z=iso["umap_z"],
                        mode="markers",
                        name="Isolated marker (X)",
                        marker=dict(
                            size=6,
                            symbol="x",
                            color=iso_color,
                            line=dict(width=0),
                        ),
                        customdata=iso_customdata,
                        hoverinfo="none",
                        hovertemplate=None,
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=iso["umap_x"],
                        y=iso["umap_y"],
                        mode="markers",
                        name=f"Isolated  (n={len(iso)})",
                        marker=dict(
                            size=20,
                            symbol="circle",
                            color=transparent,
                            line=dict(width=2.5, color=iso_color),
                        ),
                        customdata=iso_customdata,
                        hoverinfo="none",
                        hovertemplate=None,
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=iso["umap_x"],
                        y=iso["umap_y"],
                        mode="markers",
                        name="Isolated marker (X)",
                        marker=dict(
                            size=10,
                            symbol="x-thin",
                            color=iso_color,
                            line=dict(width=2, color=iso_color),
                        ),
                        customdata=iso_customdata,
                        hoverinfo="none",
                        hovertemplate=None,
                        showlegend=False,
                    )
                )

    if is_3d:
        fig.update_layout(
            title="Topic landscape (3D UMAP projection of abstract embeddings)",
            height=760,
            margin=dict(l=0, r=0, t=60, b=0),
            scene=dict(
                xaxis_title="UMAP-1",
                yaxis_title="UMAP-2",
                zaxis_title="UMAP-3",
                bgcolor="#fafafa",
            ),
            hovermode="closest",
            showlegend=False,
        )
    else:
        fig.update_layout(
            title="Topic landscape (UMAP projection of abstract embeddings)",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            height=760,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="#fafafa",
            # `hovermode='closest'` ensures plotly_hover fires for the nearest
            # point even though no on-canvas tooltip is shown.
            hovermode="closest",
            # The plotly legend is replaced by a custom HTML legend rendered in
            # the sidebar (see save_figure), so the plot itself stays uncluttered.
            showlegend=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    return fig


_INFO_PANEL_CSS = """
html, body { margin: 0; padding: 0; height: 100%; background: #fff; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       color: #222; }
#wrap { display: flex; flex-direction: row; width: 100%; height: 100vh;
        min-height: 760px; box-sizing: border-box;
        padding: 12px; gap: 12px; }
#plot-host { flex: 1 1 auto; position: relative; min-width: 0; height: 100%; }
#plot-host > div { width: 100% !important; height: 100% !important; }
#sidebar { flex: 0 0 340px; height: 100%; box-sizing: border-box;
           display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.card { background: #fafafa; border: 1px solid #d4d4d4; border-radius: 6px;
        box-sizing: border-box; }
#legend { flex: 0 1 auto; max-height: 48%; overflow-y: auto;
          padding: 10px 12px; font-size: 12.5px; line-height: 1.35; }
#legend h4 { margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase;
             letter-spacing: 0.04em; color: #666; font-weight: 600; }
#legend .item { display: flex; align-items: center; gap: 8px;
                padding: 3px 6px; border-radius: 4px; cursor: pointer;
                user-select: none; }
#legend .item:hover { background: #eef1f5; }
#legend .item.hidden { opacity: 0.4; }
#legend .item.hidden .name { text-decoration: line-through; }
#legend .item svg { flex: 0 0 auto; }
#legend .item .name { word-break: break-word; }
#info-panel { flex: 1 1 auto; min-height: 120px; overflow-y: auto;
              padding: 14px 16px;
              font-size: 13px; line-height: 1.5; }
#info-panel.empty { color: #999; font-style: italic; }
#info-panel h3 { font-size: 14px; margin: 0 0 8px 0; line-height: 1.35; }
#info-panel .meta { color: #555; font-size: 12px; margin-bottom: 10px; }
#info-panel .pill { display: inline-block; padding: 1px 6px;
                    background: #eef1f5; border-radius: 3px;
                    margin: 0 4px 2px 0; }
#info-panel p { margin: 8px 0; }
#info-panel a { color: #2563eb; text-decoration: none; word-break: break-all; }
#info-panel a:hover { text-decoration: underline; }
@media (max-width: 900px) {
  #wrap { flex-direction: column; height: auto; }
  #plot-host { height: 65vh; min-height: 480px; }
  #sidebar { flex: 0 0 auto; height: auto; }
  #legend { max-height: 30vh; }
  #info-panel { max-height: 40vh; }
}
"""


_HOVER_PANEL_JS = """
(function() {
  function init() {
    var plot = document.querySelector('#plot-host .plotly-graph-div');
    var info = document.getElementById('info-panel');
    var legend = document.getElementById('legend');
    if (!plot || !info || !legend || !plot.on || !window.Plotly) {
      setTimeout(init, 50);
      return;
    }
    function escapeHtml(s) {
      if (s == null) return '';
      return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // Hide the in-plot legend so the figure can use its full canvas area.
    Plotly.relayout(plot, {showlegend: false});

    // --- Custom HTML legend -------------------------------------------------
    // Small SVG glyphs that approximate the plotly marker symbols we use.
    // The mapping is intentionally compact: anything unknown falls back to
    // a filled circle so we never crash on a missing entry.
    var SYMBOL_PATHS = {
      'circle':         '<circle cx="7" cy="7" r="5"/>',
      'square':         '<rect x="2" y="2" width="10" height="10"/>',
      'diamond':        '<polygon points="7,1 13,7 7,13 1,7"/>',
      'diamond-tall':   '<polygon points="7,1 12,7 7,13 2,7"/>',
      'triangle-up':    '<polygon points="7,2 13,12 1,12"/>',
      'triangle-down':  '<polygon points="7,12 13,2 1,2"/>',
      'triangle-left':  '<polygon points="2,7 12,1 12,13"/>',
      'triangle-right': '<polygon points="12,7 2,1 2,13"/>',
      'star':           '<polygon points="7,1 8.6,5.3 13,5.5 9.5,8.3 10.8,12.6 7,10.1 3.2,12.6 4.5,8.3 1,5.5 5.4,5.3"/>',
      'pentagon':       '<polygon points="7,1.5 13,5.6 10.7,12.5 3.3,12.5 1,5.6"/>',
      'hexagon':        '<polygon points="7,1 12.5,4 12.5,10 7,13 1.5,10 1.5,4"/>',
      'hexagram':       '<polygon points="7,1 8.5,5 13,5 9.7,7.7 11,12.5 7,10 3,12.5 4.3,7.7 1,5 5.5,5"/>',
      'octagon':        '<polygon points="4.5,1.5 9.5,1.5 12.5,4.5 12.5,9.5 9.5,12.5 4.5,12.5 1.5,9.5 1.5,4.5"/>',
      'cross':          '<polygon points="5,1 9,1 9,5 13,5 13,9 9,9 9,13 5,13 5,9 1,9 1,5 5,5"/>',
      'hourglass':      '<polygon points="2,2 12,2 7,7 12,12 2,12 7,7"/>',
      'bowtie':         '<polygon points="2,2 12,12 12,2 2,12 2,2"/>',
      'star-square':    '<rect x="2" y="2" width="10" height="10"/><circle cx="7" cy="7" r="2" fill="white"/>',
      'star-diamond':   '<polygon points="7,1 13,7 7,13 1,7"/><circle cx="7" cy="7" r="2" fill="white"/>'
    };

    function isTransparent(c) {
      if (!c) return false;
      var s = String(c).replace(/\\s+/g, '').toLowerCase();
      if (s === 'transparent' || s === 'none') return true;
      // rgba(...,0) or rgba(...,0.0): last component is the alpha
      var m = s.match(/^rgba?\\(([^)]+)\\)$/);
      if (m) {
        var parts = m[1].split(',');
        if (parts.length === 4 && parseFloat(parts[3]) === 0) return true;
      }
      return false;
    }
    // Returns just the inner SVG shapes (no <svg> wrapper) so callers can
    // stack multiple markers in a single composite SVG.
    function symbolGlyph(symbol, marker) {
      var raw = String(symbol || 'circle');
      var isOpen = raw.indexOf('-open') !== -1;
      var base = raw.replace('-open', '').replace('-dot', '');
      var fillColor = (marker && marker.color) || '#888';
      var lineColor = (marker && marker.line && marker.line.color) || '#333';
      var ringLike = isOpen || isTransparent(fillColor);
      if (base === 'x-thin' || base === 'x') {
        var c = ringLike ? lineColor : fillColor;
        return '<g stroke="' + c + '" stroke-width="2" '
          + 'stroke-linecap="round" fill="none">'
          + '<line x1="3" y1="3" x2="11" y2="11"/>'
          + '<line x1="3" y1="11" x2="11" y2="3"/>'
          + '</g>';
      }
      var inner = SYMBOL_PATHS[base] || SYMBOL_PATHS['circle'];
      var fill = ringLike ? 'none' : fillColor;
      var stroke = ringLike ? lineColor : '#333';
      var strokeWidth = ringLike ? 2 : 1;
      return '<g fill="' + fill + '" stroke="' + stroke + '" '
        + 'stroke-width="' + strokeWidth + '">' + inner + '</g>';
    }
    function symbolSvg(layers /* array of {symbol, marker} */) {
      var arr = Array.isArray(layers) ? layers : [layers];
      var inner = arr.map(function(l) {
        return symbolGlyph(l.symbol, l.marker);
      }).join('');
      return '<svg viewBox="0 0 14 14" width="14" height="14">'
        + inner + '</svg>';
    }

    function buildLegend() {
      legend.innerHTML = '<h4>Topics (click to toggle)</h4>';
      // Build groups: each legend-visible trace is a primary; any
      // immediately-following trace with showlegend:false is treated as a
      // companion overlay (e.g. the inner X under the Isolated ring) and
      // shares the same legend entry + toggle behaviour.
      var groups = [];
      (plot.data || []).forEach(function(tr, idx) {
        if (tr.showlegend === false) {
          if (groups.length) {
            groups[groups.length - 1].layerIdx.push(idx);
          }
        } else {
          groups.push({primaryIdx: idx, layerIdx: [idx]});
        }
      });
      groups.forEach(function(g) {
        var primary = plot.data[g.primaryIdx];
        var layers = g.layerIdx.map(function(i) {
          var d = plot.data[i];
          return {symbol: d.marker && d.marker.symbol, marker: d.marker};
        });
        var item = document.createElement('div');
        item.className = 'item';
        item.dataset.allIdx = JSON.stringify(g.layerIdx);
        if (primary.visible === 'legendonly' || primary.visible === false) {
          item.classList.add('hidden');
        }
        item.innerHTML = symbolSvg(layers) +
          '<span class="name">' + escapeHtml(primary.name || ('trace ' + g.primaryIdx)) + '</span>';
        item.addEventListener('click', function() {
          var idxs = JSON.parse(item.dataset.allIdx);
          var cur = plot.data[idxs[0]].visible;
          var nextVisible = (cur === false || cur === 'legendonly') ? true : 'legendonly';
          Plotly.restyle(plot, {visible: nextVisible}, idxs).then(function() {
            item.classList.toggle('hidden', nextVisible !== true);
          });
        });
        legend.appendChild(item);
      });
    }

    buildLegend();

    // --- Hover -> info panel ------------------------------------------------
    function render(pt) {
      var cd = pt.customdata || [];
      var title = cd[0] || '(no title)';
      var year = cd[1] || '';
      var cites = (cd[2] === '' || cd[2] == null) ? '' : (cd[2] + ' citations');
      var abstract = cd[3] || '';
      var doi = cd[4] || '';
      var topic = (pt.data && pt.data.name) ? pt.data.name : '';
      var pills = [];
      if (year) pills.push('<span class="pill">' + escapeHtml(year) + '</span>');
      if (cites) pills.push('<span class="pill">' + escapeHtml(cites) + '</span>');
      if (topic) pills.push('<span class="pill">' + escapeHtml(topic) + '</span>');
      var doiHtml = doi
        ? '<p><a href="https://doi.org/' + escapeHtml(doi) + '" target="_blank" rel="noopener">doi.org/' + escapeHtml(doi) + '</a></p>'
        : '';
      info.classList.remove('empty');
      info.innerHTML =
        '<h3>' + escapeHtml(title) + '</h3>' +
        '<div class="meta">' + pills.join(' ') + '</div>' +
        '<p>' + escapeHtml(abstract) + '</p>' +
        doiHtml;
    }
    plot.on('plotly_hover', function(data) {
      if (!data || !data.points || !data.points.length) return;
      render(data.points[0]);
    });
    // No `plotly_unhover` clearing on purpose: the last hovered paper stays
    // visible so the user can read it without keeping the cursor steady.
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
"""


def _uses_sidebar(fig: go.Figure) -> bool:
    """Whether the landscape sidebar (custom legend + info panel) should wrap
    the figure. Detection is based on our landscape convention: traces have
    `customdata` set and the in-plot tooltip is suppressed (`hoverinfo='none'`)."""
    return any(
        getattr(tr, "hoverinfo", None) == "none" and getattr(tr, "customdata", None) is not None
        for tr in fig.data
    )


def landscape_html(fig: go.Figure) -> str:
    """Build the self-contained landscape HTML document as a string.

    Returns either the figure + sidebar wrapper (when the figure is a
    landscape produced by :func:`landscape_figure`) or, as a fallback,
    Plotly's own `fig.to_html(...)` full document.

    The returned string is a complete `<!doctype html>...</html>` document
    that can be saved to disk via :func:`save_figure` or embedded inline in
    a notebook via :func:`show_landscape`.
    """
    if not _uses_sidebar(fig):
        return fig.to_html(include_plotlyjs="cdn", full_html=True)

    # The info panel lives in a sibling flex column (outside the plot area),
    # so we don't need to reserve plot real estate for it. Just let the
    # figure fill its container.
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        autosize=True,
        height=None,
    )

    plot_div = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="landscape-plot",
        config={"responsive": True},
    )

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Topic landscape</title>"
        f"<style>{_INFO_PANEL_CSS}</style></head><body>"
        "<div id=\"wrap\">"
        f"<div id=\"plot-host\">{plot_div}</div>"
        "<aside id=\"sidebar\">"
        "<div id=\"legend\" class=\"card\"></div>"
        "<div id=\"info-panel\" class=\"card empty\">"
        "Hover over any point in the plot to see paper details here."
        "</div>"
        "</aside>"
        "</div>"
        f"<script>{_HOVER_PANEL_JS}</script>"
        "</body></html>"
    )


def save_figure(fig: go.Figure, path: str | Path) -> Path:
    """Write the landscape figure as a self-contained HTML file.

    Thin wrapper around :func:`landscape_html` that just persists the
    returned document to ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(landscape_html(fig), encoding="utf-8")
    return path


def show_landscape(fig: go.Figure, height: int = 780):
    """Display the landscape inline in a Jupyter / Colab notebook.

    The full HTML document (Plotly chart + sidebar legend + info panel) is
    embedded in an isolated ``<iframe srcdoc=...>``. This sidesteps two
    common pitfalls:

    * In Colab, relative ``<iframe src="...">`` paths cannot reach files on
      disk (the cell output runs in a sandbox), so loading a saved
      ``landscape.html`` shows a blank frame.
    * ``IPython.display.HTML`` injected directly into the notebook would let
      our sidebar CSS / JS leak into the host page.

    Falls back to ``fig.show()`` if IPython is unavailable.
    """
    try:
        from IPython.display import HTML, display
    except ImportError:
        fig.show()
        return None

    import html as _html
    doc = landscape_html(fig)
    encoded = _html.escape(doc, quote=True)
    iframe = (
        f'<iframe srcdoc="{encoded}" '
        f'style="width:100%; height:{int(height)}px; border:1px solid #ddd; '
        f'border-radius:6px;" '
        f'sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox"'
        f'></iframe>'
    )
    return display(HTML(iframe))


def _render_table(title: str, df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return f"<h2>{html.escape(title)}</h2><p><em>No rows.</em></p>"
    rows: list[str] = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{html.escape(str(row[c]) if pd.notna(row[c]) else '')}</td>"
            for c in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    return (
        f"<h2>{html.escape(title)}</h2>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


REPORT_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 24px; color: #222; max-width: 1200px; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 32px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left;
         vertical-align: top; }
th { background: #f5f5f5; }
tr:nth-child(even) { background: #fafafa; }
iframe { border: none; width: 100%; height: 760px; }
.meta { color: #666; font-size: 13px; }
"""


def build_report(
    *,
    query: str,
    n_papers: int,
    figure_path: Path,
    topic_info: pd.DataFrame,
    isolated_df: pd.DataFrame,
    output_path: str | Path,
    top_isolated: int = 25,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_rel = Path(figure_path).name

    topic_table = topic_info.copy()
    if "Topic" in topic_table.columns:
        topic_table = topic_table.rename(
            columns={"Topic": "topic", "Count": "n_papers", "Name": "name"}
        )
        topic_table = topic_table[["topic", "n_papers", "name"]]

    iso_columns = [
        c
        for c in ["title", "year", "venue", "citation_count", "isolation_score", "doi"]
        if c in isolated_df.columns
    ]
    iso_view = isolated_df.head(top_isolated)[iso_columns]

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Research scan: {html.escape(query)}</title>
<style>{REPORT_CSS}</style></head>
<body>
<h1>Research scan: {html.escape(query)}</h1>
<p class="meta">{n_papers} papers &middot; topic model + isolation analysis</p>

<h2>Topic landscape</h2>
<iframe src="{html.escape(figure_rel)}"></iframe>

{_render_table("Topics", topic_table, list(topic_table.columns))}
{_render_table(f"Top {top_isolated} isolated papers", iso_view, iso_columns)}
</body></html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
