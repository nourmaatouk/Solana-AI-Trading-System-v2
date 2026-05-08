from __future__ import annotations

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from realtime_predictor import (
    CONFIDENCE_THRESHOLD,
    FEATURE_COLUMNS_PATH,
    FEATURE_PATH,
    HORIZON,
    MODEL_PATH,
    SCALER_PATH,
    align_latest_features,
    latest_orderbook_unavailable,
    load_expected_features,
    score_to_signal,
    validate_artifact_contract,
)


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "realtime_classifier_dashboard.html"


def _fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _signal_for_row(row: pd.Series, expected_features: list[str]) -> tuple[str, str]:
    if "orderbook_data_available" in row.index:
        orderbook_missing = latest_orderbook_unavailable(pd.DataFrame([row]), expected_features)
        if orderbook_missing:
            return "NEUTRAL", "ORDERBOOK_UNAVAILABLE"
    return score_to_signal(float(row["prob_up"]))


def score_feature_rows(
    df: pd.DataFrame,
    expected_features: list[str],
    scaler,
    model: xgb.XGBClassifier,
) -> pd.DataFrame:
    working = df.copy()
    missing_features = [col for col in expected_features if col not in working.columns]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Realtime feature file is missing trained features: {missing}")

    feature_frame = working.loc[:, expected_features].replace([np.inf, -np.inf], np.nan)
    valid_mask = feature_frame.notna().all(axis=1)
    valid_rows = working.loc[valid_mask].copy()
    valid_features = feature_frame.loc[valid_mask]

    if valid_rows.empty:
        raise ValueError("No rows have a complete trained feature set.")

    probabilities = model.predict_proba(scaler.transform(valid_features.to_numpy(dtype=np.float64)))[:, 1]
    valid_rows["prob_up"] = probabilities
    valid_rows["prob_down"] = 1.0 - valid_rows["prob_up"]

    decisions = valid_rows.apply(lambda row: _signal_for_row(row, expected_features), axis=1)
    valid_rows["signal"] = [decision[0] for decision in decisions]
    valid_rows["decision_reason"] = [decision[1] for decision in decisions]

    return valid_rows


def _line_svg(
    values: list[float],
    labels: list[str],
    *,
    width: int = 980,
    height: int = 280,
    y_min: float = 0.0,
    y_max: float = 1.0,
    thresholds: tuple[float, float] | None = None,
    stroke: str = "#2563eb",
) -> str:
    if not values:
        return "<svg viewBox='0 0 980 280' role='img'></svg>"

    pad_left = 52
    pad_right = 24
    pad_top = 24
    pad_bottom = 42
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_for(index: int) -> float:
        if len(values) == 1:
            return pad_left + plot_w
        return pad_left + (index / (len(values) - 1)) * plot_w

    def y_for(value: float) -> float:
        clipped = min(max(float(value), y_min), y_max)
        return pad_top + (1.0 - ((clipped - y_min) / (y_max - y_min))) * plot_h

    points = [(x_for(i), y_for(v)) for i, v in enumerate(values)]
    path = " ".join(
        f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}"
        for i, (x, y) in enumerate(points)
    )
    grid_lines = []
    for mark in [y_min, (y_min + y_max) / 2, y_max]:
        y = y_for(mark)
        grid_lines.append(
            f"<line x1='{pad_left}' y1='{y:.2f}' x2='{width - pad_right}' y2='{y:.2f}' class='chart-grid' />"
            f"<text x='{pad_left - 10}' y='{y + 4:.2f}' text-anchor='end' class='axis'>{_fmt(mark, 2)}</text>"
        )

    threshold_lines = []
    if thresholds:
        for value in thresholds:
            y = y_for(value)
            threshold_lines.append(
                f"<line x1='{pad_left}' y1='{y:.2f}' x2='{width - pad_right}' y2='{y:.2f}' class='threshold' />"
                f"<text x='{width - pad_right - 4}' y='{y - 6:.2f}' text-anchor='end' class='threshold-label'>{_fmt(value, 2)}</text>"
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

    circles = []
    for x, y in points[-12:]:
        circles.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.5' class='point' />")

    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Model probability chart">
      <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
      {''.join(grid_lines)}
      {''.join(threshold_lines)}
      <path d="{path}" fill="none" stroke="{stroke}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
      {''.join(circles)}
      {''.join(ticks)}
    </svg>
    """


def build_html_report(scored: pd.DataFrame, expected_features: list[str], row_limit: int) -> str:
    display_rows = scored.tail(row_limit).copy()
    latest = display_rows.iloc[-1]

    labels = [
        str(ts)[5:16]
        for ts in display_rows.get("timestamp", pd.Series(range(len(display_rows)))).tolist()
    ]
    prob_values = display_rows["prob_up"].astype(float).tolist()
    close_values = display_rows["close"].astype(float).tolist() if "close" in display_rows else []

    close_min = min(close_values) if close_values else 0.0
    close_max = max(close_values) if close_values else 1.0
    close_pad = max((close_max - close_min) * 0.12, 0.5)

    orderbook_available = int(latest.get("orderbook_data_available", 1)) == 1
    signal = html.escape(str(latest["signal"]))
    reason = html.escape(str(latest["decision_reason"]))
    signal_class = signal.lower()

    rows_html = []
    for _, row in display_rows.tail(12).iterrows():
        row_signal = html.escape(str(row["signal"]))
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('timestamp', 'n/a')))}</td>"
            f"<td>{_fmt(row.get('close', np.nan), 2)}</td>"
            f"<td>{_fmt(row['prob_up'], 4)}</td>"
            f"<td>{_fmt(row['prob_down'], 4)}</td>"
            f"<td><span class='pill {row_signal.lower()}'>{row_signal}</span></td>"
            f"<td>{html.escape(str(row['decision_reason']))}</td>"
            f"<td>{int(row.get('orderbook_data_available', 1))}</td>"
            "</tr>"
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    feature_count = len(expected_features)
    chart_prob = _line_svg(
        prob_values,
        labels,
        thresholds=(CONFIDENCE_THRESHOLD, 1.0 - CONFIDENCE_THRESHOLD),
        stroke="#2563eb",
    )
    chart_close = _line_svg(
        close_values,
        labels,
        y_min=close_min - close_pad,
        y_max=close_max + close_pad,
        stroke="#0f766e",
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SOL Classification Model Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #111827;
      --muted: #667085;
      --line: #d7dce4;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --up: #047857;
      --down: #b42318;
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
    .span-2 {{ grid-column: span 2; }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-grid {{ stroke: #e8ebf0; }}
    .axis {{ fill: #667085; font-size: 12px; }}
    .tick {{ stroke: #98a2b3; }}
    .threshold {{ stroke: #f59e0b; stroke-width: 1.6; stroke-dasharray: 6 6; }}
    .threshold-label {{ fill: #b45309; font-size: 12px; font-weight: 700; }}
    .point {{ fill: white; stroke: #2563eb; stroke-width: 2; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-width: 76px;
      justify-content: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .pill.up, .metric-value.up {{ color: var(--up); }}
    .pill.down, .metric-value.down {{ color: var(--down); }}
    .pill.neutral, .metric-value.neutral {{ color: var(--neutral); }}
    .pill.up {{ background: #ecfdf3; }}
    .pill.down {{ background: #fef3f2; }}
    .pill.neutral {{ background: #f2f4f7; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid #edf0f4;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
    }}
    @media (max-width: 760px) {{
      header {{ display: block; }}
      .grid {{ grid-template-columns: 1fr; }}
      .span-2 {{ grid-column: span 1; }}
      .table-wrap {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>SOL Classification Model Dashboard</h1>
        <div class="muted">H{HORIZON} classifier, {HORIZON} x 15m candles. Generated {generated_at}.</div>
      </div>
      <div class="muted">Source: {html.escape(str(FEATURE_PATH.name))}</div>
    </header>

    <section class="grid" aria-label="Latest signal metrics">
      <div class="panel">
        <div class="metric-label">Final Signal</div>
        <div class="metric-value {signal_class}">{signal}</div>
        <div class="muted">{reason}</div>
      </div>
      <div class="panel">
        <div class="metric-label">Probability Up</div>
        <div class="metric-value">{_fmt(latest["prob_up"], 4)}</div>
        <div class="muted">Gate up >= {_fmt(CONFIDENCE_THRESHOLD, 2)}</div>
      </div>
      <div class="panel">
        <div class="metric-label">Probability Down</div>
        <div class="metric-value">{_fmt(latest["prob_down"], 4)}</div>
        <div class="muted">Gate down >= {_fmt(CONFIDENCE_THRESHOLD, 2)}</div>
      </div>
      <div class="panel">
        <div class="metric-label">Latest Close</div>
        <div class="metric-value">{_fmt(latest.get("close", np.nan), 2)}</div>
        <div class="muted">{html.escape(str(latest.get("timestamp", "n/a")))}</div>
      </div>
    </section>

    <section class="grid" aria-label="Data checks">
      <div class="panel">
        <div class="metric-label">Feature Contract</div>
        <div class="metric-value">{feature_count}/{feature_count}</div>
        <div class="muted">Model, scaler, and registry matched</div>
      </div>
      <div class="panel">
        <div class="metric-label">Rows Scored</div>
        <div class="metric-value">{len(scored)}</div>
        <div class="muted">Complete rows only</div>
      </div>
      <div class="panel">
        <div class="metric-label">Orderbook Latest</div>
        <div class="metric-value">{'OK' if orderbook_available else 'MISSING'}</div>
        <div class="muted">Fresh bucket required for trades</div>
      </div>
      <div class="panel">
        <div class="metric-label">Decision Rule</div>
        <div class="metric-value">{_fmt(1.0 - CONFIDENCE_THRESHOLD, 2)} / {_fmt(CONFIDENCE_THRESHOLD, 2)}</div>
        <div class="muted">Neutral inside the band</div>
      </div>
    </section>

    <section class="panel" style="margin-bottom: 14px;">
      <h2>Probability Up</h2>
      {chart_prob}
    </section>

    <section class="panel" style="margin-bottom: 14px;">
      <h2>Close Price</h2>
      {chart_close}
    </section>

    <section class="panel">
      <h2>Recent Scored Candles</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Close</th>
              <th>Prob Up</th>
              <th>Prob Down</th>
              <th>Signal</th>
              <th>Reason</th>
              <th>Orderbook</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def generate_dashboard(output_path: Path, row_limit: int) -> pd.DataFrame:
    expected_features = load_expected_features(FEATURE_COLUMNS_PATH)
    df = pd.read_csv(FEATURE_PATH)
    if df.empty:
        raise ValueError(f"Realtime feature file is empty: {FEATURE_PATH}")

    scaler = joblib.load(SCALER_PATH)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    validate_artifact_contract(expected_features, scaler, model)

    align_latest_features(df, expected_features)
    scored = score_feature_rows(df, expected_features, scaler, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html_report(scored, expected_features, row_limit), encoding="utf-8")

    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small HTML dashboard for the realtime XGBoost classifier.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the HTML report.")
    parser.add_argument("--rows", type=int, default=80, help="Number of recent scored rows to plot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scored = generate_dashboard(args.output, max(5, args.rows))
    latest = scored.iloc[-1]

    print(f"Dashboard saved to: {args.output}")
    print(f"Latest signal: {latest['signal']} ({latest['decision_reason']})")
    print(f"Probability UP: {latest['prob_up']:.4f}")
    print(f"Probability DOWN: {latest['prob_down']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
