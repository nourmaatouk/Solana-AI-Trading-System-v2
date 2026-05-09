import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from src.evaluation.metrics import compute_detection_delay_ms
from src.config import N_FEATURES_KAGGLE, N_FEATURES_LOBSTER

logger = logging.getLogger(__name__)


def run_flash_crash_sim(
    output_json: Path,
    model: Optional[object] = None,
    device: Optional[torch.device] = None,
    adjacency: Optional[torch.Tensor] = None,
    n_assets: int = 1,
    d_input: int = 33,
) -> None:
    """Simulate NAVSTRAT layering strategy and measure model detection delay.

    Timeline (all times in ms):
        0  – 2000  : normal benign trading
        2000 – 4000: NAVSTRAT sell-side layering (spoofing active)
        4000+      : liquidity vacuum / crash

    Predictions come from the supplied model. If ``adjacency`` is provided the
    model is called as ``model(X, adjacency)`` (TEN-GNN path). Otherwise it is
    called as ``model(X)`` (single-asset TEN path). If no model is provided the
    function produces a clearly-labelled SIMULATION-ONLY result.

    Parameters
    ----------
    output_json : Path to write results JSON.
    model       : Optional trained TENSpoofingDetector or TENGNNModel.
    device      : torch device (defaults to CPU).
    adjacency   : (n_assets, n_assets) tensor for TEN-GNN mode.
    n_assets    : number of assets (only used when adjacency is provided).
    d_input     : feature dimension (default 33, matching Kaggle checkpoint).
    """
    logger.info("Running Flash Crash NAVSTRAT Layering Simulator...")

    spoofing_start_ms = 2000.0
    crash_time_ms = 4000.0
    event_times_ms = np.arange(0, 5000, 50)  # evaluate every 50 ms
    n_steps = len(event_times_ms)

    if device is None:
        if model is not None:
            # Inherit the device the model already lives on
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    if model is not None:
        # ── Real model inference ──────────────────────────────────────────────
        # Inject phase-specific features into normalized z-score space.
        # Three informative indices (approximate positions in 33-feature schema):
        #   feature ~0: midpoint (not used directly here)
        #   feature ~1: spread
        #   feature ~2: order imbalance
        # For 49-feature LOBSTER schema: indices 43=delta_t, 44=imbalance, 45=spread.
        # We use relative indices that work for both schemas.
        seq_len = 100
        rng = np.random.default_rng(42)

        # Resolve feature indices from the known schema.
        # Kaggle 33-feature: midpoint=0, spread=1, bid/ask notionals interleaved 2..31, imbalance=32.
        # LOBSTER 49-feature: delta_t=43, imbalance=44, spread=45.
        # The previous d_input-k heuristic mapped spread_idx→asks_notional_15 and
        # delta_t_idx→bids_notional_15, injecting into deep book columns instead of
        # the intended microstructure features.
        if d_input == N_FEATURES_KAGGLE:
            imbalance_idx    = 32
            cancel_ratio_idx = 33
            spread_idx       = 1
            midpoint_idx     = 0
            delta_t_idx      = None  # not present in Kaggle schema
        elif d_input == N_FEATURES_LOBSTER:
            imbalance_idx    = 44
            cancel_ratio_idx = None  # not a named feature in LOBSTER schema
            spread_idx       = 45
            delta_t_idx      = 43
            midpoint_idx     = 46
        else:
            logger.warning(
                "Unrecognised d_input=%d; falling back to last-3 heuristic. "
                "Injection may hit wrong features.", d_input
            )
            imbalance_idx    = d_input - 1
            cancel_ratio_idx = None
            spread_idx       = d_input - 2
            delta_t_idx      = d_input - 3 if d_input >= 3 else 0
            midpoint_idx     = None

        model.eval()
        predictions = np.zeros(n_steps, dtype=float)

        if device is None:
            device = torch.device("cpu")
        if adjacency is not None:
            adjacency = adjacency.to(device)

        for step_i, t in enumerate(event_times_ms):
            X = np.zeros((1, seq_len, d_input), dtype=np.float32)

            if t < spoofing_start_ms:
                # Benign phase: near-zero imbalance, tight spread, stable midpoint
                X[0, :, imbalance_idx] = rng.normal(0.0, 0.1, seq_len)
                X[0, :, spread_idx]    = rng.uniform(0.01, 0.05, seq_len)
                if delta_t_idx is not None:
                    X[0, :, delta_t_idx] = rng.exponential(1.0, seq_len)
                if midpoint_idx is not None:
                    X[0, :, midpoint_idx] = np.cumsum(rng.normal(0, 0.01, seq_len))
            elif t < crash_time_ms:
                # Spoofing phase: one-sided imbalance spike, high cancel ratio,
                # spread compression from apparent depth illusion.
                X[0, :, imbalance_idx] = rng.uniform(-2.0, -1.5, seq_len)
                X[0, :, spread_idx]    = rng.uniform(-1.5, -1.0, seq_len)
                if cancel_ratio_idx is not None:
                    X[0, :, cancel_ratio_idx] = rng.uniform(2.0, 3.5, seq_len)
                if delta_t_idx is not None:
                    X[0, :, delta_t_idx] = rng.uniform(-1.0, -0.5, seq_len)
                if midpoint_idx is not None:
                    X[0, :, midpoint_idx] = np.cumsum(rng.normal(-0.02, 0.01, seq_len))
            else:
                # Vacuum/crash phase: extreme imbalance, wide spread
                X[0, :, imbalance_idx] = rng.uniform(-3.0, -2.5, seq_len)
                X[0, :, spread_idx]    = rng.uniform(2.0, 3.0, seq_len)
                if delta_t_idx is not None:
                    X[0, :, delta_t_idx] = rng.exponential(0.5, seq_len)
                if midpoint_idx is not None:
                    X[0, :, midpoint_idx] = np.cumsum(rng.normal(-0.05, 0.02, seq_len))

            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                if adjacency is not None:
                    # TEN-GNN: expand to (B, N, T, D)
                    X_gnn = X_tensor.unsqueeze(1).expand(-1, n_assets, -1, -1)
                    logits = model(X_gnn, adjacency)
                else:
                    # Single-asset TEN
                    logits = model(X_tensor)
            proba = torch.softmax(logits, dim=-1)[0, 1].item()
            predictions[step_i] = proba

        simulation_mode = "real_model_inference"

    else:
        # ── Simulation-only mode (no real model) ──────────────────────────────
        # This is a deterministic ramp used only to test the metrics pipeline.
        # The detection delay here is SCRIPTED, NOT MEASURED.
        logger.warning(
            "No model provided — using scripted simulation predictions. "
            "The detection delay is NOT a real measurement. "
            "Pass a trained model to get meaningful results."
        )
        rng = np.random.default_rng(42)
        predictions = np.zeros(n_steps, dtype=float)
        for i, t in enumerate(event_times_ms):
            if t < spoofing_start_ms:
                predictions[i] = rng.uniform(0.01, 0.15)
            elif t < crash_time_ms:
                time_since_spoof = t - spoofing_start_ms
                confidence = min(0.98, 0.1 + time_since_spoof / 250.0)
                predictions[i] = float(np.clip(confidence + rng.uniform(-0.05, 0.05), 0, 1))
            else:
                predictions[i] = 0.99

        simulation_mode = "scripted_simulation_only_not_real_detection"

    delay_ms = compute_detection_delay_ms(event_times_ms, predictions, spoofing_start_ms)
    early_warning_ms = (crash_time_ms - spoofing_start_ms) - delay_ms if not np.isnan(delay_ms) else float("nan")

    results = {
        "scenario": "NAVSTRAT Layering to Liquidity Vacuum",
        "simulation_mode": simulation_mode,
        "spoofing_start_ms": spoofing_start_ms,
        "crash_time_ms": crash_time_ms,
        "detection_delay_ms": float(delay_ms),
        "early_warning_lead_time_ms": float(early_warning_ms),
        "status": "PASS" if (not np.isnan(early_warning_ms) and early_warning_ms > 0) else "FAIL",
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    existing_data: dict = {}
    if output_json.exists():
        with open(output_json, "r") as f:
            existing_data = json.load(f)

    existing_data["flash_crash_simulator"] = results

    with open(output_json, "w") as f:
        json.dump(existing_data, f, indent=4)

    print("\n" + "=" * 70)
    print("FLASH CRASH SIMULATOR: NAVSTRAT LAYERING")
    print("=" * 70)
    print(f"Mode:                 {simulation_mode}")
    print(f"Spoofing Injected at: {spoofing_start_ms:.1f} ms")
    print(f"Crash / Vacuum Event: {crash_time_ms:.1f} ms")
    print(f"Detection Delay:      {delay_ms:.1f} ms" if not np.isnan(delay_ms) else "Detection Delay:      never detected")
    print(f"Early Warning Lead:   {early_warning_ms:.1f} ms" if not np.isnan(early_warning_ms) else "Early Warning Lead:   N/A")
    print(f"System Response:      {results['status']}")
    print("=" * 70)
    if simulation_mode != "real_model_inference":
        print("NOTE: results above use scripted predictions, not a trained model.")
    logger.info("Saved Flash Crash simulation results to %s", output_json)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import torch
    root = Path(__file__).resolve().parent.parent.parent
    out_path = root / "results" / "evaluation_report.json"
    ckpt_path = root / "checkpoints" / "tengnn_btc_eth_sol_1min" / "best_model.pt"

    _model, _adj, _n_assets, _d_input = None, None, 3, 33
    if ckpt_path.exists():
        from src.deployment.inference_engine import SpoofingInferenceEngine
        engine = SpoofingInferenceEngine.from_checkpoint(ckpt_path)
        _model = engine._model
        _adj   = engine._default_adj
        _n_assets = engine._model.n_assets
        _d_input  = engine._model.d_input

    run_flash_crash_sim(
        out_path,
        model=_model,
        adjacency=_adj,
        n_assets=_n_assets,
        d_input=_d_input,
    )
