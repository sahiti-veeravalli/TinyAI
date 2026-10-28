"""
Teacher Model Training for TinyAI.
Provides:
1. PyTorch Tabular Deep MLP (ResNet-style skip connections, LayerNorm, Swish)
2. High-capacity LightGBM / XGBoost ensemble
3. Export to standardized ONNX graph for downstream pruning/quantization.
"""

import os
import io
import time
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TabularMLPTeacher(nn.Module if HAS_TORCH else object):
    """
    High-capacity Tabular Deep MLP with residual blocks.
    Represents the uncompressed enterprise cloud model (large memory footprint, high accuracy).
    """
    def __init__(self, in_features: int, hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TabularMLPTeacher")
        super().__init__()
        self.in_features = in_features
        
        # Layer 1
        self.fc1 = nn.Linear(in_features, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        
        # Layer 2 (Residual block)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.res_proj = nn.Linear(hidden_dims[0], hidden_dims[1]) if hidden_dims[0] != hidden_dims[1] else nn.Identity()

        # Layer 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.ln3 = nn.LayerNorm(hidden_dims[2])
        self.act3 = nn.SiLU()

        # Head
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.drop1(self.act1(self.ln1(self.fc1(x))))
        h2 = self.drop2(self.act2(self.ln2(self.fc2(h1)))) + self.res_proj(h1)
        h3 = self.act3(self.ln3(self.fc3(h2)))
        logits = self.head(h3)
        return logits.squeeze(-1)


class TeacherPipeline:
    """Trains, evaluates, and exports the high-accuracy teacher model."""

    def __init__(self, in_features: int, model_type: str = "mlp_resnet", hidden_dims: Tuple[int, ...] = (256, 128, 64)):
        self.in_features = in_features
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.model: Optional[Any] = None
        self.device = torch.device("cpu") if HAS_TORCH else "cpu"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 1e-3,
        class_weight_ratio: float = 1.0,
        progress_cb: Optional[Callable[[int, int, float, float], None]] = None
    ) -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not yet loaded in environment.")

        self.model = TabularMLPTeacher(self.in_features, hidden_dims=self.hidden_dims).to(self.device)
        
        # Positional weight for class imbalance
        pos_weight = torch.tensor([class_weight_ratio], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        history = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_y)
            scheduler.step()

            # Validation evaluation
            val_probs = self.predict_proba(X_val)
            val_auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.5
            val_loss = total_loss / len(X_train)

            history.append({"epoch": epoch, "loss": val_loss, "val_auc": val_auc})
            if progress_cb:
                progress_cb(epoch, epochs, val_loss, val_auc)

        return {"history": history, "final_val_auc": val_auc}

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or not HAS_TORCH:
            raise RuntimeError("Model not fitted.")
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).float().to(self.device)
            logits = self.model(tensor_x).cpu().numpy()
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def export_onnx(self, output_path: str) -> str:
        """Exports PyTorch teacher to standard ONNX format."""
        if not HAS_TORCH or self.model is None:
            raise RuntimeError("Model not trained or torch missing.")
        self.model.eval()
        dummy_input = torch.randn(1, self.in_features, dtype=torch.float32, device=self.device)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False
        )
        return output_path

    def get_size_bytes(self) -> int:
        """Computes in-memory parameter footprint in bytes."""
        if self.model is None:
            return 0
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total_bytes
