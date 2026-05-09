import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.deployment.inference_engine import compile_torchscript_model
from src.deployment.quantize import quantize_deployment_safe_modules
from src.models.ten_gnn import TENGNNModel


def build_small_ten_gnn() -> TENGNNModel:
    return TENGNNModel(
        n_assets=2,
        d_input=5,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    ).eval()


def test_ten_gnn_compiles_to_torchscript():
    model = build_small_ten_gnn()
    sequences = torch.randn(1, 2, 8, 5)
    adjacency = torch.eye(2)

    compiled = compile_torchscript_model(model, (sequences, adjacency))
    logits = compiled(sequences, adjacency)

    assert logits.shape == (1, 2)


def test_deployment_safe_quantized_model_keeps_forward_shape():
    model = build_small_ten_gnn()
    quantized = quantize_deployment_safe_modules(model)
    sequences = torch.randn(1, 2, 8, 5)
    adjacency = torch.eye(2)

    logits = quantized(sequences, adjacency)

    assert logits.shape == (1, 2)
