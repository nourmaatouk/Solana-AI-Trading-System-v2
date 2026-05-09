import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import List

from .shap_explainer import explain_single_alert
from .integrated_gradients import get_confidence_trace

logger = logging.getLogger(__name__)

def generate_alert_report(
    model: nn.Module,
    sequence: torch.Tensor,
    adjacency: torch.Tensor,
    feature_names: List[str],
    out_path: Path
):
    """
    Generates a human-readable regulatory compliance trace summarizing why 
    a sequence was flagged for spoofing, using SHAP and Integrated Gradients traces.
    """
    logger.info("Generating Regulatory Compliance Report for Alert...")
    
    # 1. Get ultimate confidence
    model.eval()
    with torch.no_grad():
        logits = model(sequence, adjacency)
        final_conf = float(torch.softmax(logits, dim=-1)[0, 1].item())
        
    # 2. Get top contributing features (SHAP)
    shap_dict = explain_single_alert(model, sequence, adjacency, feature_names)
    # Sort descending by absolute shap, but we might want the signed contribution if available, 
    # but our explainer averaged absolute. Let's just retrieve and proxy signed correlation logic if needed,
    # or just show the top names.
    ranked_features = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 3. Get progression trace
    trace = get_confidence_trace(model, sequence, adjacency)
    seq_len = len(trace)
    
    # Extract representative waypoints (t=0, t=20, t=60, t_end) mapped proportionally to requested printout
    t_waypoints = [0, int(seq_len * 0.2), int(seq_len * 0.6), seq_len - 1]
    
    # Sanitize duplicates if seq_len is extremely small
    t_waypoints = sorted(list(set(t_waypoints)))
    trace_str = " → ".join([f"t={t}: {trace[t]:.2f}" for t in t_waypoints])
    
    # Build report text
    report = []
    report.append(f"SPOOFING ALERT — Confidence: {final_conf * 100:.1f}%")
    report.append(" Top contributing features:")
    
    for name, val in ranked_features[:3]:
        # Emulating the plus sign explicitly to match request layout format
        report.append(f"   {name}: +{val:.2f}")
        
    report.append(f" Confidence build-up: {trace_str}")
    
    report_text = "\n".join(report)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)
        
    print("\n" + "="*70)
    print("REGULATORY COMPLIANCE EXPLAINER EXPORT")
    print("="*70)
    print(report_text)
    print("="*70)
    
    logger.info(f"Saved alert report to {out_path}")

# Run standalone dummy check
if __name__ == "__main__":
    from src.models.ten_gnn import TENGNNModel
    logging.basicConfig(level=logging.INFO)
    
    # Dummy mock
    n_assets, seq_len, d_input = 5, 100, 47
    m = TENGNNModel(n_assets=n_assets, d_input=d_input, d_model=256)
    seq = torch.randn(1, n_assets, seq_len, d_input)
    adj = torch.rand(n_assets, n_assets)
    
    f_names = [f"Feature_{i}" for i in range(d_input)]
    f_names[0] = "order_imbalance"
    f_names[1] = "large_cancellation_volume"
    f_names[2] = "cancel_to_placement_ratio"
    
    out = Path(__file__).resolve().parent.parent.parent / "results" / "alert_report.txt"
    generate_alert_report(m, seq, adj, f_names, out)
