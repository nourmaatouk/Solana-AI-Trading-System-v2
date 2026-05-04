import torch
from pathlib import Path
from src.models.ten_gnn import TENGNNModel

def save_dummy_weights():
    root = Path(__file__).resolve().parent.parent.parent
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model = TENGNNModel(n_assets=5)
    torch.save(model.state_dict(), checkpoint_dir / "ten_gnn_latest.pt")
    print(f"Saved dummy weights to {checkpoint_dir / 'ten_gnn_latest.pt'}")

if __name__ == "__main__":
    save_dummy_weights()
