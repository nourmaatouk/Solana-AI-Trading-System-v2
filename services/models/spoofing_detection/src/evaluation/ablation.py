import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.evaluation.metrics import compute_point_metrics
from src.models.ten import TENSpoofingDetector, AdaptiveTemporalPE
from src.models.ten_gnn import TENGNNModel

logger = logging.getLogger(__name__)


class TENGNNNoAdaptivePE(TENGNNModel):
    """TEN-GNN with adaptive temporal PE disabled (uses standard sinusoidal PE).

    Overrides the TEN encoder's positional encoding to ignore delta_t,
    giving a fair comparison for the ablation study.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace adaptive PE with a version that always ignores delta_t
        orig_pe = self.ten_encoder.positional_encoding

        class StandardPE(AdaptiveTemporalPE):
            def forward(self, x, delta_t=None):
                return super().forward(x, delta_t=None)  # always ignore delta_t

        self.ten_encoder.positional_encoding = StandardPE(
            d_model=orig_pe.d_model,
            max_len=orig_pe.max_len,
            dropout=orig_pe.dropout.p,
        )


def run_ablation(output_json: Path, n_samples: int = 100):
    """Evaluates 4 architectural variants of TEN-GNN (ablation study).

    NOTE: All models are randomly initialised — this measures latency and
    architectural differences (parameter count, forward-pass structure), not
    trained accuracy. Load a real checkpoint before reporting F1 numbers.
    """
    logger.info("Running Ablation Study for TEN-GNN (random-weight models)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 100
    n_assets = 3
    d_input_full = 33
    d_input_no_micro = 10

    torch.manual_seed(42)
    X_full = torch.randn(n_samples, n_assets, seq_len, d_input_full).to(device)
    X_no_micro = torch.randn(n_samples, n_assets, seq_len, d_input_no_micro).to(device)
    dummy_adj = torch.rand(n_assets, n_assets).to(device)

    y_test = torch.randint(0, 2, (n_samples,)).cpu().numpy()

    results = {}

    def eval_variant(name: str, model_fn, input_tensor, is_gnn: bool = True):
        model = model_fn().to(device)
        model.eval()

        with torch.no_grad():
            if is_gnn:
                model(input_tensor[:2], dummy_adj)
            else:
                model(input_tensor[:2])

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()

            if is_gnn:
                logits = model(input_tensor, dummy_adj)
            else:
                logits = model(input_tensor)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            lat_ms = (time.perf_counter() - t0) * 1000 / n_samples

        preds = logits.argmax(dim=-1).cpu().numpy()
        m = compute_point_metrics(y_test, preds)
        m["latency_ms_per_seq"] = lat_ms
        m["note"] = "random-weight baseline — train model before comparing F1"
        m["n_params"] = sum(p.numel() for p in model.parameters())
        results[name] = m

    # 1. Full TEN-GNN
    eval_variant(
        "Full TEN-GNN",
        lambda: TENGNNModel(n_assets=n_assets, d_input=d_input_full),
        X_full, is_gnn=True,
    )

    # 2. w/o GNN — use only the target-asset TEN embedding (no cross-asset context)
    X_target_only = X_full[:, 0, :, :]
    eval_variant(
        "w/o GNN (TEN only)",
        lambda: TENSpoofingDetector(d_input=d_input_full, d_model=256),
        X_target_only, is_gnn=False,
    )

    # 3. w/o adaptive PE — same architecture but PE ignores delta_t (standard sinusoidal)
    eval_variant(
        "w/o adaptive PE (Standard Sinusoidal)",
        lambda: TENGNNNoAdaptivePE(n_assets=n_assets, d_input=d_input_full),
        X_full, is_gnn=True,
    )

    # 4. w/o microstructure features — reduced input dimension (price/vol only, no micro)
    eval_variant(
        "w/o microstructure features",
        lambda: TENGNNModel(n_assets=n_assets, d_input=d_input_no_micro),
        X_no_micro, is_gnn=True,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    existing_data = {}
    if output_json.exists():
        with open(output_json, "r") as f:
            existing_data = json.load(f)

    existing_data["ablation"] = results

    with open(output_json, "w") as f:
        json.dump(existing_data, f, indent=4)

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS  (random-weight models — latency only)")
    print(f"{'Variant':<40} | {'F1':<6} | {'Params':>10} | {'Latency_ms':<10}")
    print("-" * 70)
    for model_name, m in results.items():
        print(
            f"{model_name:<40} | {m['f1']:.4f} | {m['n_params']:>10,} | {m['latency_ms_per_seq']:.4f}"
        )
    print("=" * 70)
    print("NOTE: F1 values above are from untrained random-weight models and are")
    print("      meaningless. Run with a trained checkpoint to get real F1 numbers.")
    logger.info("Saved ablation results to %s", output_json)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out_path = Path(__file__).resolve().parent.parent.parent / "results" / "evaluation_report.json"
    run_ablation(out_path)
