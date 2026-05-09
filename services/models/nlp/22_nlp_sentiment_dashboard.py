from __future__ import annotations

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "nlp_sentiment_dashboard.html"

PROCESSED_ARTICLES_PATH = BASE_DIR / "data" / "processed" / "processed_articles.csv"
SENTIMENT_FEATURES_PATH = BASE_DIR / "data" / "processed" / "sentiment_features_15m.csv"
REALTIME_SENTIMENT_PATH = (
    BASE_DIR.parent / "classification" / "data" / "realtime" / "latest_sentiment_15m.csv"
)


def _fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _int_fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return "0"
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "0"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalize_time(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame.columns:
        return frame
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")
    return normalized


def _line_svg(
    values: list[float],
    labels: list[str],
    *,
    width: int = 980,
    height: int = 280,
    y_min: float | None = None,
    y_max: float | None = None,
    zero_line: bool = False,
    stroke: str = "#2563eb",
) -> str:
    if not values:
        return "<div class='empty-chart'>No sentiment buckets available.</div>"

    clean_values = [float(v) for v in values if not pd.isna(v)]
    if not clean_values:
        return "<div class='empty-chart'>No numeric sentiment values available.</div>"

    if y_min is None:
        y_min = min(clean_values)
    if y_max is None:
        y_max = max(clean_values)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    pad_left = 52
    pad_right = 24
    pad_top = 24
    pad_bottom = 42
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_for(index: int) -> float:
        if len(values) == 1:
            return pad_left + plot_w
        return pad_left + (index / (len(values) - 1)) * plot_w

    def y_for(value: float) -> float:
        clipped = min(max(float(value), y_min), y_max)
        return pad_top + (1.0 - ((clipped - y_min) / (y_max - y_min))) * plot_h

    points = [(x_for(i), y_for(float(value))) for i, value in enumerate(values)]
    path = " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )

    grid_lines = []
    for mark in [y_min, (y_min + y_max) / 2, y_max]:
        y = y_for(mark)
        grid_lines.append(
            f"<line x1='{pad_left}' y1='{y:.2f}' x2='{width - pad_right}' y2='{y:.2f}' class='chart-grid' />"
            f"<text x='{pad_left - 10}' y='{y + 4:.2f}' text-anchor='end' class='axis'>{_fmt(mark, 2)}</text>"
        )

    zero_markup = ""
    if zero_line and y_min < 0 < y_max:
        y_zero = y_for(0.0)
        zero_markup = (
            f"<line x1='{pad_left}' y1='{y_zero:.2f}' x2='{width - pad_right}' y2='{y_zero:.2f}' "
            "class='zero-line' />"
        )

    tick_positions = sorted(set([0, len(values) // 2, len(values) - 1]))
    ticks = []
    for index in tick_positions:
        x = x_for(index)
        label = html.escape(labels[index])
        ticks.append(
            f"<line x1='{x:.2f}' y1='{height - pad_bottom}' x2='{x:.2f}' y2='{height - pad_bottom + 6}' class='tick' />"
            f"<text x='{x:.2f}' y='{height - 14}' text-anchor='middle' class='axis'>{label}</text>"
        )

    circles = [
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.5' class='point' />"
        for x, y in points[-12:]
    ]

    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Sentiment line chart">
      <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
      {''.join(grid_lines)}
      {zero_markup}
      <path d="{path}" fill="none" stroke="{stroke}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
      {''.join(circles)}
      {''.join(ticks)}
    </svg>
    """


def _bar_svg(
    labels: list[str],
    values: list[float],
    *,
    width: int = 980,
    height: int = 260,
    fill: str = "#0f766e",
) -> str:
    if not labels or not values:
        return "<div class='empty-chart'>No article distribution available.</div>"

    max_value = max(max(values), 1.0)
    pad_left = 150
    pad_right = 28
    pad_top = 18
    row_h = 32
    chart_h = max(height, pad_top + len(values) * row_h + 20)
    plot_w = width - pad_left - pad_right

    rows = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = pad_top + index * row_h
        bar_w = (float(value) / max_value) * plot_w
        rows.append(
            f"<text x='{pad_left - 10}' y='{y + 18}' text-anchor='end' class='axis'>{html.escape(label)}</text>"
            f"<rect x='{pad_left}' y='{y + 5}' width='{bar_w:.2f}' height='18' rx='4' fill='{fill}' />"
            f"<text x='{pad_left + bar_w + 8:.2f}' y='{y + 19}' class='axis'>{_int_fmt(value)}</text>"
        )

    return f"""
    <svg class="chart" viewBox="0 0 {width} {chart_h}" role="img" aria-label="Article distribution">
      <rect x="0" y="0" width="{width}" height="{chart_h}" fill="white" />
      {''.join(rows)}
    </svg>
    """


def _status_table(paths: Iterable[Path]) -> str:
    rows = []
    for path in paths:
        status = "found" if path.exists() else "missing"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(path.relative_to(BASE_DIR.parent.parent.parent)))}</td>"
            f"<td><span class='status {status}'>{status.upper()}</span></td>"
            "</tr>"
        )
    return "".join(rows)


def _latest_sentiment_row(sentiment_df: pd.DataFrame, realtime_df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if not realtime_df.empty:
        return realtime_df.iloc[-1], "realtime bridge"
    if not sentiment_df.empty:
        return sentiment_df.iloc[-1], "offline NLP"
    return None, "missing"


def build_html_report(
    sentiment_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    realtime_df: pd.DataFrame,
    row_limit: int,
) -> str:
    sentiment_df = _normalize_time(sentiment_df)
    articles_df = _normalize_time(articles_df)
    realtime_df = _normalize_time(realtime_df)

    latest, latest_source = _latest_sentiment_row(sentiment_df, realtime_df)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if latest is None:
        latest_timestamp = "n/a"
        sentiment_mean = "n/a"
        news_volume = "0"
        positive_ratio = "n/a"
        negative_ratio = "n/a"
        high_impact = "0"
        sentiment_class = "neutral"
    else:
        latest_timestamp = html.escape(str(latest.get("timestamp", "n/a")))
        latest_score = float(latest.get("sentiment_mean", 0.0) or 0.0)
        sentiment_mean = _fmt(latest_score, 4)
        news_volume = _int_fmt(latest.get("news_volume", 0))
        positive_ratio = _fmt(latest.get("positive_ratio", 0.0), 3)
        negative_ratio = _fmt(latest.get("negative_ratio", 0.0), 3)
        high_impact = _int_fmt(latest.get("high_impact_event_count", latest.get("high_impact_event", 0)))
        sentiment_class = "positive" if latest_score > 0.1 else "negative" if latest_score < -0.1 else "neutral"

    chart_frame = sentiment_df.tail(row_limit).copy()
    labels = []
    if not chart_frame.empty and "timestamp" in chart_frame.columns:
        labels = [str(ts)[5:16] for ts in chart_frame["timestamp"].tolist()]
    sentiment_values = (
        chart_frame["sentiment_mean"].astype(float).fillna(0.0).tolist()
        if "sentiment_mean" in chart_frame.columns
        else []
    )
    volume_values = (
        chart_frame["news_volume"].astype(float).fillna(0.0).tolist()
        if "news_volume" in chart_frame.columns
        else []
    )

    sentiment_chart = _line_svg(
        sentiment_values,
        labels,
        y_min=-1.0,
        y_max=1.0,
        zero_line=True,
        stroke="#2563eb",
    )
    volume_chart = _line_svg(
        volume_values,
        labels,
        y_min=0.0,
        y_max=max(volume_values) + 1.0 if volume_values else 1.0,
        stroke="#0f766e",
    )

    event_chart = "<div class='empty-chart'>No processed article rows available.</div>"
    if not articles_df.empty and "event_type" in articles_df.columns:
        event_counts = articles_df["event_type"].fillna("unknown").value_counts().head(8)
        event_chart = _bar_svg(event_counts.index.astype(str).tolist(), event_counts.astype(float).tolist())

    bucket_rows = []
    for _, row in chart_frame.tail(12).iterrows():
        bucket_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('timestamp', 'n/a')))}</td>"
            f"<td>{_fmt(row.get('sentiment_mean', np.nan), 4)}</td>"
            f"<td>{_int_fmt(row.get('news_volume', 0))}</td>"
            f"<td>{_fmt(row.get('positive_ratio', np.nan), 3)}</td>"
            f"<td>{_fmt(row.get('negative_ratio', np.nan), 3)}</td>"
            f"<td>{_fmt(row.get('event_signal_sum', np.nan), 4)}</td>"
            f"<td>{_fmt(row.get('sentiment_spike', np.nan), 4)}</td>"
            "</tr>"
        )

    article_rows = []
    if not articles_df.empty:
        working_articles = articles_df.copy()
        if "event_signal" in working_articles.columns:
            working_articles["_abs_signal"] = working_articles["event_signal"].astype(float).abs()
            working_articles = working_articles.sort_values("_abs_signal", ascending=False)
        else:
            working_articles["_abs_signal"] = 0.0

        for _, row in working_articles.head(12).iterrows():
            title = html.escape(str(row.get("title", "n/a")))
            if len(title) > 110:
                title = title[:107] + "..."
            article_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('timestamp', 'n/a')))}</td>"
                f"<td>{title}</td>"
                f"<td>{html.escape(str(row.get('event_type', 'general')))}</td>"
                f"<td>{_fmt(row.get('sentiment_score', np.nan), 4)}</td>"
                f"<td>{_fmt(row.get('event_signal', np.nan), 4)}</td>"
                f"<td>{html.escape(str(row.get('sentiment_label', 'neutral')))}</td>"
                "</tr>"
            )

    if not bucket_rows:
        bucket_rows.append("<tr><td colspan='7'>No sentiment bucket rows found.</td></tr>")
    if not article_rows:
        article_rows.append("<tr><td colspan='6'>No processed article rows found.</td></tr>")

    status_rows = _status_table(
        [SENTIMENT_FEATURES_PATH, PROCESSED_ARTICLES_PATH, REALTIME_SENTIMENT_PATH]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NLP Sentiment Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #111827;
      --muted: #667085;
      --line: #d7dce4;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --positive: #047857;
      --negative: #b42318;
      --neutral: #475467;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 16px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .span-2 {{ grid-column: span 2; }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric-value {{
      margin-top: 6px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .positive {{ color: var(--positive); }}
    .negative {{ color: var(--negative); }}
    .neutral {{ color: var(--neutral); }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-grid {{ stroke: #e8ebf0; }}
    .axis {{ fill: #667085; font-size: 12px; }}
    .tick {{ stroke: #98a2b3; }}
    .zero-line {{ stroke: #f59e0b; stroke-width: 1.4; stroke-dasharray: 6 6; }}
    .point {{ fill: white; stroke: #2563eb; stroke-width: 2; }}
    .empty-chart {{
      min-height: 180px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border: 1px dashed #c8ced8;
      border-radius: 8px;
      background: #fbfcfd;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .status.found {{ color: #047857; background: #ecfdf3; }}
    .status.missing {{ color: #b42318; background: #fef3f2; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid #edf0f4;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
      white-space: nowrap;
    }}
    .table-wrap {{ overflow-x: auto; }}
    @media (max-width: 760px) {{
      header {{ display: block; }}
      .grid {{ grid-template-columns: 1fr; }}
      .span-2 {{ grid-column: span 1; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>NLP Sentiment Dashboard</h1>
        <div class="muted">15-minute sentiment features generated from GDELT and RSS. Generated {generated_at}.</div>
      </div>
      <div class="muted">Latest source: {html.escape(latest_source)}</div>
    </header>

    <section class="grid" aria-label="Latest NLP metrics">
      <div class="panel">
        <div class="metric-label">Latest Sentiment</div>
        <div class="metric-value {sentiment_class}">{sentiment_mean}</div>
        <div class="muted">{latest_timestamp}</div>
      </div>
      <div class="panel">
        <div class="metric-label">News Volume</div>
        <div class="metric-value">{news_volume}</div>
        <div class="muted">Articles in latest bucket</div>
      </div>
      <div class="panel">
        <div class="metric-label">Positive / Negative</div>
        <div class="metric-value">{positive_ratio} / {negative_ratio}</div>
        <div class="muted">Ratios in latest bucket</div>
      </div>
      <div class="panel">
        <div class="metric-label">High Impact</div>
        <div class="metric-value">{high_impact}</div>
        <div class="muted">High-impact event count</div>
      </div>
    </section>

    <section class="grid" aria-label="NLP charts">
      <div class="panel span-2">
        <h2>Sentiment Mean</h2>
        {sentiment_chart}
      </div>
      <div class="panel span-2">
        <h2>News Volume</h2>
        {volume_chart}
      </div>
    </section>

    <section class="grid" aria-label="NLP distributions">
      <div class="panel span-2">
        <h2>Event Distribution</h2>
        {event_chart}
      </div>
      <div class="panel span-2">
        <h2>Data Files</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Path</th><th>Status</th></tr>
            </thead>
            <tbody>{status_rows}</tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="panel" style="margin-bottom: 14px;">
      <h2>Recent Sentiment Buckets</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Sentiment</th>
              <th>Volume</th>
              <th>Positive</th>
              <th>Negative</th>
              <th>Event Signal</th>
              <th>Spike</th>
            </tr>
          </thead>
          <tbody>{''.join(bucket_rows)}</tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Top Article Signals</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Title</th>
              <th>Event</th>
              <th>Sentiment</th>
              <th>Signal</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody>{''.join(article_rows)}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def generate_dashboard(output_path: Path, row_limit: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sentiment_df = _read_csv(SENTIMENT_FEATURES_PATH)
    articles_df = _read_csv(PROCESSED_ARTICLES_PATH)
    realtime_df = _read_csv(REALTIME_SENTIMENT_PATH)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html_report(sentiment_df, articles_df, realtime_df, row_limit),
        encoding="utf-8",
    )

    return sentiment_df, articles_df, realtime_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML dashboard for NLP sentiment results.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the HTML report.")
    parser.add_argument("--rows", type=int, default=80, help="Number of recent sentiment buckets to plot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sentiment_df, articles_df, realtime_df = generate_dashboard(args.output, max(5, args.rows))
    latest, source = _latest_sentiment_row(_normalize_time(sentiment_df), _normalize_time(realtime_df))

    print(f"NLP dashboard saved to: {args.output}")
    print(f"Sentiment buckets: {len(sentiment_df)}")
    print(f"Processed articles: {len(articles_df)}")
    if latest is not None:
        print(f"Latest source: {source}")
        print(f"Latest timestamp: {latest.get('timestamp')}")
        print(f"Latest sentiment_mean: {_fmt(latest.get('sentiment_mean'))}")
        print(f"Latest news_volume: {_int_fmt(latest.get('news_volume'))}")
    else:
        print("Latest source: missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
