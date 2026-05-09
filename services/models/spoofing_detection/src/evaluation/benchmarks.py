import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import compute_point_metrics
from src.models.ten import TENSpoofingDetector
from src.models.ten_gnn import TENGNNModel

logger = logging.getLogger(__name__)


# ── 1. LSTM-Attention Baseline ───────────────────────────────────────────────
class LSTMAttention(nn.Module):
    """2-layer BiLSTM + dot-product attention baseline."""

    def __init__(self, d_input: int = 49, hidden_size: int = 256,
                 num_layers: int = 2, n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_input, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x, adj=None):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)


# ── 2. Rule-based Baseline ───────────────────────────────────────────────────
def rule_based_detector(order_imbalance: np.ndarray, delta_t_ms: np.ndarray) -> np.ndarray:
    """Flag a sequence as spoofing if order imbalance peaks above 0.7 AND a burst
    of fast events (< 10 ms) occurs anywhere in the sequence.

    Uses feature indices from the LOBSTER 49-feature schema (synthetic data path):
        index 44 — order_imbalance = (bid_vol - ask_vol) / total_vol
        index 43 — delta_t_ms (time since last event)

    Both inputs should be the PEAK statistic over the full sequence:
        order_imbalance = max(|imbalance|) over all timesteps
        delta_t_ms      = min(delta_t)     over all timesteps

    Checking only the last timestep misses the layering phase (t=30–60) where
    spoofing signals are injected; the aftermath phase (t=80–100) resets both
    features to benign values.
    """
    return ((np.abs(order_imbalance) > 0.7) & (delta_t_ms < 10.0)).astype(int)


# ── 3. Evaluation Runner ─────────────────────────────────────────────────────
def run_benchmarks(
    output_json: Path,
    n_samples_synth: int = 1000,
) -> None:
    """Evaluate baseline and deep models on synthetic data.

    Classical ML models (Logistic Regression, Random Forest) are trained on the
    synthetic train split — their F1 numbers are real.

    Deep learning models (LSTM, TEN, TEN-GNN) use random weights here — their
    F1 numbers are meaningless. Load a trained checkpoint before comparing F1.
    Latency numbers are real on all models.
    """
    from src.data.dataset import prepare_data

    logger.info("Running benchmarks (Synthetic: %d sequences)...", n_samples_synth)

    # NOTE: Real LOBSTER data (AAPL) path is disabled — it requires external
    # data files not present in this repo. Only the synthetic path runs here.
    data_synth = prepare_data(use_real_data=False, n_sequences=n_samples_synth)

    all_results = []

    def evaluate_on_source(source_name: str, datasets: dict) -> None:
        logger.info("Evaluating on %s data...", source_name)
        test_ds = datasets["test"]
        X_test = test_ds.X.numpy()
        y_test = test_ds.y.numpy()
        n_feat = X_test.shape[2]

        # ── Rule-based ────────────────────────────────────────────────────────
        # Indices 44 (imbalance) and 43 (delta_t_ms) are LOBSTER-schema positions.
        # Only valid on 49-feature synthetic data; skip gracefully on 33-feature Kaggle.
        # Use PEAK |imbalance| and MINIMUM delta_t over the full sequence — the
        # spoofing signature lives in the layering phase (t=30..60), not the last step.
        t0 = time.perf_counter()
        if n_feat >= 45:
            peak_imbalance = np.abs(X_test[:, :, 44]).max(axis=1)
            min_delta_t    = X_test[:, :, 43].min(axis=1)
        else:
            peak_imbalance = np.zeros(len(y_test))
            min_delta_t    = np.full(len(y_test), np.inf)
        y_pred_rule = rule_based_detector(peak_imbalance, min_delta_t)
        lat_rule = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m_rule = compute_point_metrics(y_test, y_pred_rule)
        m_rule.update({"Model": "Rule-Based", "Data Source": source_name,
                       "latency_ms_per_seq": lat_rule, "note": "uses imbalance+delta_t"})
        all_results.append(m_rule)

        # ── Classical ML (mean-pooled features) ───────────────────────────────
        X_train_ml = datasets["train"].X.mean(dim=1).numpy()
        y_train_ml = datasets["train"].y.numpy()
        X_test_ml = X_test.mean(axis=1)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_ml, y_train_ml)
        t0 = time.perf_counter()
        y_pred_lr = lr.predict(X_test_ml)
        lat_lr = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m_lr = compute_point_metrics(y_test, y_pred_lr)
        m_lr.update({"Model": "Logistic Regression", "Data Source": source_name,
                     "latency_ms_per_seq": lat_lr})
        all_results.append(m_lr)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_ml, y_train_ml)
        t0 = time.perf_counter()
        y_pred_rf = rf.predict(X_test_ml)
        lat_rf = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m_rf = compute_point_metrics(y_test, y_pred_rf)
        m_rf.update({"Model": "Random Forest", "Data Source": source_name,
                     "latency_ms_per_seq": lat_rf})
        all_results.append(m_rf)

        # ── Deep models (random weights — latency measurement only) ───────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random_weight_note = "random-weight model — train before comparing F1"

        X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

        lstm = LSTMAttention(d_input=n_feat).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = lstm(X_tensor)
        lat_lstm = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m_lstm = compute_point_metrics(y_test, logits.argmax(dim=-1).cpu().numpy())
        m_lstm.update({"Model": "LSTM-Attention", "Data Source": source_name,
                       "latency_ms_per_seq": lat_lstm, "note": random_weight_note})
        all_results.append(m_lstm)

        ten = TENSpoofingDetector(d_input=n_feat).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = ten(X_tensor)
        lat_ten = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m_ten = compute_point_metrics(y_test, logits.argmax(dim=-1).cpu().numpy())
        m_ten.update({"Model": "TEN", "Data Source": source_name,
                      "latency_ms_per_seq": lat_ten, "note": random_weight_note})
        all_results.append(m_ten)

        # TEN-GNN: replicate single asset across 5 nodes for latency measurement.
        # For real cross-asset evaluation, supply 5 distinct asset sequences.
        tengnn = TENGNNModel(n_assets=5, d_input=n_feat).to(device)
        X_gnn = X_tensor.unsqueeze(1).expand(-1, 5, -1, -1)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = tengnn(X_gnn, torch.eye(5, device=device))
        lat_gnn = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m_gnn = compute_point_metrics(y_test, logits.argmax(dim=-1).cpu().numpy())
        m_gnn.update({"Model": "TEN-GNN", "Data Source": source_name,
                      "latency_ms_per_seq": lat_gnn,
                      "note": f"{random_weight_note}; GNN uses single-asset replication"})
        all_results.append(m_gnn)

    evaluate_on_source("Synthetic", data_synth)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    existing_data = {}
    if output_json.exists():
        with open(output_json, "r") as f:
            existing_data = json.load(f)

    existing_data["benchmarks_standardized"] = all_results

    with open(output_json, "w") as f:
        json.dump(existing_data, f, indent=4)

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS  (deep models use random weights — F1 is not meaningful until trained)")
    print("CPU latency numbers are real. Rule-based / LR / RF F1 numbers are real.")
    print("-" * 100)
    print(f"{'Model':<22} | {'Source':<10} | {'F1':<6} | {'Prec.':<7} | {'Recall':<6} | {'Latency ms':<12}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['Model']:<22} | {r['Data Source']:<10} | {r['f1']:.3f}  | "
              f"{r['precision']:.3f} | {r['recall']:.3f} | {r['latency_ms_per_seq']:.4f}")
    print("=" * 100)
    logger.info("Saved benchmark results to %s", output_json)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out_path = Path(__file__).resolve().parent.parent.parent / "results" / "evaluation_report.json"
    run_benchmarks(out_path)
