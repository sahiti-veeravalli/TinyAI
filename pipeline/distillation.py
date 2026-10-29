
"""
Knowledge Distillation Engine for TinyAI.
Transfers representational capacity from deep teacher model (or ensemble) into a
compact edge student architecture via temperature-scaled soft targets and focal loss.
"""

import os
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CompactStudentMLP(nn.Module if HAS_TORCH else object):
    """
    Ultra-lightweight edge student model.
    Configurable: default (32, 16) hidden units (~1-2KB weights).
    """
    def __init__(self, in_features: int, hidden_dims: Tuple[int, ...] = (32, 16)):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for CompactStudentMLP")
        super().__init__()
        self.in_features = in_features
        layers = []
        curr = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.SiLU())
            curr = h
        layers.append(nn.Linear(curr, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def distillation_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 3.0,
    alpha: float = 0.65,
    pos_weight: float = 1.0,
    gamma: float = 1.5  # Focal loss gamma for imbalanced edge cases
) -> torch.Tensor:
    """
    Combined Soft-target Knowledge Distillation and Focal Cross-Entropy loss.
    Math:
        L = alpha * T^2 * BCE(sigmoid(z_s / T), sigmoid(z_t / T)) + (1 - alpha) * FocalLoss(z_s, y)
    """
    # Soft target distillation
    p_t = torch.sigmoid(teacher_logits / temperature)
    p_s = torch.sigmoid(student_logits / temperature)
    soft_loss = -(p_t * torch.log(p_s + 1e-7) + (1.0 - p_t) * torch.log(1.0 - p_s + 1e-7))
    soft_loss = soft_loss.mean() * (temperature ** 2)

    # Hard task loss with focal modulation for class imbalance
    p_pred = torch.sigmoid(student_logits)
    p_t_target = targets * p_pred + (1.0 - targets) * (1.0 - p_pred)
    focal_weight = torch.pow(1.0 - p_t_target + 1e-7, gamma)
    
    bce = nn.functional.binary_cross_entropy_with_logits(
        student_logits, targets, pos_weight=torch.tensor(pos_weight, device=student_logits.device), reduction="none"
    )
    hard_loss = (focal_weight * bce).mean()

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


class DistillationEngine:
    """Coordinates distillation from teacher predictions to compact edge student."""

    def __init__(self, in_features: int, hidden_dims: Tuple[int, ...] = (32, 16)):
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.student: Optional[CompactStudentMLP] = None
        self.device = torch.device("cpu") if HAS_TORCH else "cpu"

    def distill(
        self,
        X_train: np.ndarray,
        teacher_train_logits: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        temperature: float = 3.5,
        alpha: float = 0.70,
        pos_weight: float = 1.0,
        epochs: int = 12,
        batch_size: int = 128,
        lr: float = 2e-3,
        progress_cb: Optional[Callable[[int, int, float, float], None]] = None
    ) -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required.")

        self.student = CompactStudentMLP(self.in_features, hidden_dims=self.hidden_dims).to(self.device)
        optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(teacher_train_logits).float(),
            torch.from_numpy(y_train).float()
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        history = []
        for epoch in range(1, epochs + 1):
            self.student.train()
            total_loss = 0.0
            for bx, bt_logits, by in train_loader:
                bx, bt_logits, by = bx.to(self.device), bt_logits.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                bs_logits = self.student(bx)
                loss = distillation_loss_fn(
                    bs_logits, bt_logits, by,
                    temperature=temperature, alpha=alpha, pos_weight=pos_weight
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(by)
            scheduler.step()

            # Validation metric
            val_probs = self.predict_proba(X_val)
            val_auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.5
            val_loss = total_loss / len(X_train)

            history.append({"epoch": epoch, "loss": val_loss, "val_auc": val_auc})
            if progress_cb:
                progress_cb(epoch, epochs, val_loss, val_auc)

        return {"history": history, "final_val_auc": val_auc}

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.student is None or not HAS_TORCH:
            raise RuntimeError("Student model not trained.")
        self.student.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).float().to(self.device)
            return self.student(tensor_x).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def export_onnx(self, output_path: str) -> str:
        """Exports student to ONNX format."""
        if not HAS_TORCH or self.student is None:
            raise RuntimeError("Student model not trained.")
        self.student.eval()
        dummy_input = torch.randn(1, self.in_features, dtype=torch.float32, device=self.device)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.onnx.export(
            self.student,
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
