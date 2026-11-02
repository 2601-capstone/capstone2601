"""
Hybrid CNN-LSTM Architecture for Federated Healthcare Analytics Simulation.

This module implements a Hybrid CNN-LSTM neural network and comparative baselines
for patient risk prediction using longitudinal health measurement sequences:
1. Architectural Components:
   - 1D Convolutional Neural Network (CNN): Extracts local temporal patterns and clinical volatility
     across multi-channel physiological features.
   - Long Short-Term Memory (LSTM): Models long-term temporal dependencies and disease progression trends.
   - Fully Connected Dense Head: Aggregates sequence representations for binary risk prediction (Sigmoid).
2. Comparative Baselines (Evaluated on Identical Isolated Patient-Level Test Set):
   - Flattened MLP Baseline: Multi-layer perceptron on flattened temporal measurements.
   - Pure 1D CNN Baseline: Convolutional feature extractor + Global Average Pooling.
   - Pure LSTM Baseline: Recurrent sequence model without convolutional feature extraction.
   - Hybrid CNN-LSTM Model: Combined local temporal feature extraction and recurrent trend modeling.
3. Federated Training:
   - Sample-weighted Federated Averaging (FedAvg) over state_dict parameter tensors across 3 hospital clients.
   - Non-IID client partitions preserve patient isolation without centralizing raw sequences.
"""

import os
import sys
import copy
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.temporal_data import (
    prepare_and_save_temporal_data,
    TEMPORAL_FEATURE_COLUMNS
)


# =====================================================================
# 1. NEURAL NETWORK ARCHITECTURES
# =====================================================================

class TemporalMLPBaseline(nn.Module):
    """
    Standard MLP baseline operating on flattened temporal sequences.
    Input shape: (batch_size, sequence_length, num_features) -> flattened to (batch_size, sequence_length * num_features).
    Demonstrates model performance when temporal structure is treated as unordered flat tabular features.
    """
    def __init__(
        self,
        sequence_length: int = 12,
        num_features: int = 5,
        hidden_dim: int = 16,
        dropout_rate: float = 0.1
    ):
        super(TemporalMLPBaseline, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.input_dim = sequence_length * num_features
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.sigmoid(self.fc2(x))
        return out


class TemporalCNNModel(nn.Module):
    """
    Pure 1D CNN model for temporal feature extraction without recurrence.
    Input shape: (batch_size, sequence_length, num_features).
    Transposed to (batch_size, in_channels=5, sequence_length=12) for 1D convolution.
    Uses Adaptive Average Pooling across the time dimension followed by a linear classification head.
    """
    def __init__(
        self,
        num_features: int = 5,
        conv_filters: int = 16,
        kernel_size: int = 3,
        dropout_rate: float = 0.1
    ):
        super(TemporalCNNModel, self).__init__()
        self.num_features = num_features
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(conv_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose from (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        out = self.sigmoid(self.fc(x))
        return out


class TemporalLSTMModel(nn.Module):
    """
    Pure LSTM recurrent model without convolutional front-end.
    Input shape: (batch_size, sequence_length, num_features).
    Directly models recurrent hidden state dynamics over raw input sequences.
    """
    def __init__(
        self,
        num_features: int = 5,
        lstm_units: int = 16,
        num_layers: int = 1,
        dropout_rate: float = 0.1
    ):
        super(TemporalLSTMModel, self).__init__()
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(lstm_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, F)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (B, lstm_units)
        last_hidden = self.dropout(last_hidden)
        out = self.sigmoid(self.fc(last_hidden))
        return out


class HybridCNNLSTMModel(nn.Module):
    """
    Hybrid 1D CNN-LSTM Architecture for Healthcare Risk Prediction.

    Architecture Flow:
    1. Input: (batch_size, sequence_length=12, num_features=5)
    2. Transpose to (batch_size, 5, 12) for 1D convolution.
    3. 1D CNN Layer:
       - in_channels = 5, out_channels = 16, kernel_size = 3, padding = 1
       - Extracts local physiological fluctuations and multi-feature cross-correlations.
    4. ReLU Activation:
       - Non-linear feature mapping.
    5. Transpose back to (batch_size, sequence_length=12, channels=16) for LSTM input.
    6. LSTM Recurrent Layer:
       - input_size = 16, hidden_size = 16, batch_first = True
       - Captures long-range trajectory drift, chronic exposure, and temporal trends.
    7. Sequence Aggregation:
       - Extracts the final recurrent time-step hidden vector (batch_size, 16).
    8. Dropout Regularization (p = 0.1).
    9. Fully Connected Classification Head:
       - Linear(16, 1)
    10. Sigmoid Output:
       - Probability of high-risk classification in [0, 1].
    """
    def __init__(
        self,
        sequence_length: int = 12,
        num_features: int = 5,
        conv_filters: int = 16,
        kernel_size: int = 3,
        lstm_units: int = 16,
        lstm_layers: int = 1,
        dropout_rate: float = 0.1
    ):
        super(HybridCNNLSTMModel, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers

        # 1. 1D CNN Front-End Feature Extractor
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        # 2. LSTM Recurrent Dependency Engine
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True
        )

        # 3. Dense Classification Head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(lstm_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, T, F)
        # 1. Transpose for Conv1d: (B, F, T)
        x_conv_in = x.permute(0, 2, 1)
        x_conv_out = self.relu(self.conv1(x_conv_in))  # (B, conv_filters, T)

        # 2. Transpose for LSTM: (B, T, conv_filters)
        x_lstm_in = x_conv_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm_in)             # (B, T, lstm_units)

        # 3. Last time step representation
        last_hidden = lstm_out[:, -1, :]               # (B, lstm_units)
        last_hidden = self.dropout(last_hidden)

        # 4. Binary risk probability output
        out = self.sigmoid(self.fc(last_hidden))       # (B, 1)
        return out


# =====================================================================
# 2. MODEL INSPECTION & PARAMETER UTILITIES
# =====================================================================

def count_model_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 12, 5)
) -> Dict[str, Any]:
    """
    Generate detailed structural summary and tensor shape progression for a temporal model.
    """
    total_params = count_model_parameters(model)
    layer_info = []

    for name, param in model.named_parameters():
        layer_info.append({
            "name": name,
            "shape": list(param.shape),
            "parameters": int(param.numel()),
            "requires_grad": bool(param.requires_grad)
        })

    model_type = model.__class__.__name__
    total_bytes = total_params * 4  # float32 = 4 bytes

    return {
        "model_type": model_type,
        "input_shape": list(input_shape),
        "total_parameters": total_params,
        "parameter_bytes": total_bytes,
        "parameter_kilobytes": round(total_bytes / 1024.0, 3),
        "layers": layer_info
    }


# =====================================================================
# 3. TRAINING & EVALUATION FUNCTIONS
# =====================================================================

def set_deterministic_seed(seed: int = 42) -> None:
    """Set random seeds for PyTorch and NumPy to ensure reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_temporal_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.005,
    random_seed: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train a temporal model using Binary Cross-Entropy loss and Adam optimizer.

    Returns:
        dict:
            - "model": Trained PyTorch nn.Module
            - "history": List of per-epoch loss and validation metrics
            - "final_train_loss": float
    """
    set_deterministic_seed(random_seed)

    # Convert to PyTorch tensors
    t_X = torch.tensor(X_train, dtype=torch.float32)
    t_y = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(t_X, t_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    has_val = (X_val is not None) and (y_val is not None)

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for b_x, b_y in loader:
            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = float(np.mean(batch_losses))
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4)
        }

        if has_val:
            val_metrics = evaluate_temporal_model(model, X_val, y_val)
            epoch_record["val_loss"] = round(val_metrics["loss"], 4)
            epoch_record["val_accuracy"] = round(val_metrics["accuracy"], 4)
            epoch_record["val_roc_auc"] = round(val_metrics["roc_auc"], 4)
            epoch_record["val_f1"] = round(val_metrics["f1_score"], 4)

        history.append(epoch_record)
        if verbose and (epoch % 5 == 0 or epoch == epochs):
            val_str = f" | Val Loss: {epoch_record.get('val_loss')}" if has_val else ""
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f}{val_str}")

    return {
        "model": model,
        "history": history,
        "final_train_loss": round(history[-1]["train_loss"], 4)
    }


def evaluate_temporal_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluate a temporal PyTorch model on a holdout test partition.

    Returns:
        dict:
            - loss: float (Binary Cross-Entropy)
            - accuracy: float
            - precision: float
            - recall: float
            - f1_score: float
            - roc_auc: float
            - confusion_matrix: 2x2 list
            - probabilities: np.ndarray (1D)
            - predictions: np.ndarray (1D)
    """
    model.eval()
    t_X = torch.tensor(X_test, dtype=torch.float32)
    y_true = np.asarray(y_test).ravel().astype(int)

    with torch.no_grad():
        preds = model(t_X).numpy().ravel()

    # Clip probabilities to avoid log(0) in log_loss
    eps = 1e-7
    preds_clipped = np.clip(preds, eps, 1.0 - eps)
    loss = float(log_loss(y_true, preds_clipped))

    y_pred = (preds >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, preds))
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "probabilities": preds,
        "predictions": y_pred
    }


# =====================================================================
# 4. BASELINE COMPARISONS (MLP vs CNN vs LSTM vs CNN-LSTM)
# =====================================================================

def run_cnn_lstm_baseline_comparison(
    temporal_data_dir: str = os.path.join("data", "temporal"),
    results_dir: str = os.path.join("results", "metrics"),
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.005,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate all four temporal model architectures on identical patient-level partitions:
    1. TemporalMLPBaseline (Unordered flattened sequence representation)
    2. TemporalCNNModel (Local physiological convolution without sequential recurrence)
    3. TemporalLSTMModel (Sequential recurrence without multi-feature convolution)
    4. HybridCNNLSTMModel (Joint multi-scale convolutional pattern extraction + recurrence)

    Saves:
        - results/metrics/cnn_lstm_comparison.csv
        - results/metrics/cnn_lstm_evaluation_summary.json
        - results/metrics/cnn_lstm_training_history.csv
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1. Ensure temporal data files exist
    train_path = os.path.join(temporal_data_dir, "temporal_train.npz")
    val_path = os.path.join(temporal_data_dir, "temporal_val.npz")
    test_path = os.path.join(temporal_data_dir, "temporal_test.npz")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        prepare_and_save_temporal_data(output_dir=temporal_data_dir, random_seed=random_seed)

    train_npz = np.load(train_path, allow_pickle=True)
    val_npz = np.load(val_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)

    X_train, y_train = train_npz["X"], train_npz["y"]
    X_val, y_val = val_npz["X"], val_npz["y"]
    X_test, y_test = test_npz["X"], test_npz["y"]

    seq_len = X_train.shape[1]
    num_feat = X_train.shape[2]

    # 2. Define models
    models = {
        "MLP Baseline (Flattened)": TemporalMLPBaseline(
            sequence_length=seq_len,
            num_features=num_feat,
            hidden_dim=16
        ),
        "Pure 1D CNN": TemporalCNNModel(
            num_features=num_feat,
            conv_filters=16,
            kernel_size=3
        ),
        "Pure LSTM": TemporalLSTMModel(
            num_features=num_feat,
            lstm_units=16
        ),
        "Hybrid CNN-LSTM": HybridCNNLSTMModel(
            sequence_length=seq_len,
            num_features=num_feat,
            conv_filters=16,
            kernel_size=3,
            lstm_units=16
        )
    }

    comparison_records = []
    eval_summaries = {}
    history_dfs = []

    print("\n" + "=" * 65)
    print("PHASE 14: TEMPORAL HEALTHCARE MODEL BENCHMARK")
    print(f"Dataset: Train={len(y_train)} | Val={len(y_val)} | Test={len(y_test)} (Isolated Patients)")
    print(f"Sequence Dimensions: {seq_len} time steps x {num_feat} features")
    print("=" * 65)

    for m_name, model in models.items():
        print(f"\nTraining {m_name} ({epochs} epochs, lr={lr})...")
        t0 = time.perf_counter()
        train_res = train_temporal_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            random_seed=random_seed,
            verbose=False
        )
        elapsed = time.perf_counter() - t0

        # Evaluate on test set
        test_metrics = evaluate_temporal_model(model, X_test, y_test)
        params = count_model_parameters(model)
        size_kb = round((params * 4) / 1024.0, 3)

        # Collect training history
        hist = train_res["history"]
        for row in hist:
            history_dfs.append({
                "model": m_name,
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row.get("val_loss", None),
                "val_roc_auc": row.get("val_roc_auc", None),
                "val_f1": row.get("val_f1", None)
            })

        rec = {
            "model": m_name,
            "parameters": params,
            "size_kb": size_kb,
            "training_time_sec": round(elapsed, 3),
            "test_loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"]
        }
        comparison_records.append(rec)
        eval_summaries[m_name] = {
            "parameters": params,
            "size_kb": size_kb,
            "training_time_sec": round(elapsed, 3),
            "metrics": {k: v for k, v in test_metrics.items() if k not in ["probabilities", "predictions"]}
        }

        print(f"  {m_name:26s} | AUC: {test_metrics['roc_auc']:.4f} | F1: {test_metrics['f1_score']:.4f} | Acc: {test_metrics['accuracy']:.4f} | Params: {params}")

    df_comp = pd.DataFrame(comparison_records)
    comp_csv_path = os.path.join(results_dir, "cnn_lstm_comparison.csv")
    df_comp.to_csv(comp_csv_path, index=False)

    df_hist = pd.DataFrame(history_dfs)
    hist_csv_path = os.path.join(results_dir, "cnn_lstm_training_history.csv")
    df_hist.to_csv(hist_csv_path, index=False)

    summary_json_path = os.path.join(results_dir, "cnn_lstm_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump({
            "experiment": "Phase 14 Hybrid CNN-LSTM Benchmark",
            "sequence_length": seq_len,
            "num_features": num_feat,
            "training_configuration": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "random_seed": random_seed
            },
            "dataset_sizes": {
                "train_patients": len(y_train),
                "val_patients": len(y_val),
                "test_patients": len(y_test)
            },
            "models_evaluated": eval_summaries
        }, f, indent=4)

    print("\nBenchmark completed and artifacts saved:")
    print(f"  - Comparison Table: {comp_csv_path}")
    print(f"  - Detailed Summary: {summary_json_path}")
    print(f"  - Training History: {hist_csv_path}")
    print("=" * 65)

    return {
        "comparison_df": df_comp,
        "evaluation_summary": eval_summaries,
        "trained_models": models,
        "history_df": df_hist
    }


# =====================================================================
# 5. FEDERATED LEARNING WITH HYBRID CNN-LSTM
# =====================================================================

def aggregate_state_dicts(
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_weights: List[float]
) -> Dict[str, torch.Tensor]:
    """
    Perform Sample-Weighted Federated Averaging (FedAvg) over PyTorch state_dicts:
        W_global = sum_{k=1}^K (n_k / N) * W_k
    """
    total_weight = sum(client_weights)
    norm_weights = [w / total_weight for w in client_weights]

    aggregated_state = {}
    first_dict = client_state_dicts[0]

    for key in first_dict.keys():
        # Check if parameter tensor is floating point
        if first_dict[key].is_floating_point():
            accum = torch.zeros_like(first_dict[key])
            for k_idx, sd in enumerate(client_state_dicts):
                accum += norm_weights[k_idx] * sd[key]
            aggregated_state[key] = accum
        else:
            # For non-floating point buffers (e.g., num_batches_tracked), copy first
            aggregated_state[key] = first_dict[key].clone()

    return aggregated_state


def train_federated_cnn_lstm(
    temporal_data_dir: str = os.path.join("data", "temporal"),
    results_dir: str = os.path.join("results", "metrics"),
    num_rounds: int = 5,
    local_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 0.005,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Simulate Federated Learning of Hybrid CNN-LSTM across 3 Non-IID Hospital Clients:
    1. Distributes patient sequences across Hospital 1, Hospital 2, Hospital 3.
    2. In each round:
       - Coordinator broadcasts global CNN-LSTM weights to clients.
       - Each hospital performs local SGD/Adam optimization on its isolated sequences.
       - Coordinator aggregates client weight updates using sample-weighted FedAvg.
       - Coordinator evaluates global model on central validation and holdout test sets.
    3. Saves federated round progression to results/metrics/cnn_lstm_federated_history.csv.
    """
    os.makedirs(results_dir, exist_ok=True)
    set_deterministic_seed(random_seed)

    # 1. Load client partitions and test partitions
    client_files = {
        "hospital_1": os.path.join(temporal_data_dir, "hospital_1_temporal.npz"),
        "hospital_2": os.path.join(temporal_data_dir, "hospital_2_temporal.npz"),
        "hospital_3": os.path.join(temporal_data_dir, "hospital_3_temporal.npz")
    }
    val_path = os.path.join(temporal_data_dir, "temporal_val.npz")
    test_path = os.path.join(temporal_data_dir, "temporal_test.npz")

    if not all(os.path.exists(p) for p in list(client_files.values()) + [val_path, test_path]):
        prepare_and_save_temporal_data(output_dir=temporal_data_dir, random_seed=random_seed)

    clients_data = {}
    for c_id, path in client_files.items():
        npz = np.load(path, allow_pickle=True)
        clients_data[c_id] = {
            "X": npz["X"],
            "y": npz["y"],
            "num_samples": len(npz["y"])
        }

    val_npz = np.load(val_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)

    X_val, y_val = val_npz["X"], val_npz["y"]
    X_test, y_test = test_npz["X"], test_npz["y"]

    seq_len = X_val.shape[1]
    num_feat = X_val.shape[2]

    # Initialize Global Hybrid CNN-LSTM Model
    global_model = HybridCNNLSTMModel(
        sequence_length=seq_len,
        num_features=num_feat,
        conv_filters=16,
        kernel_size=3,
        lstm_units=16
    )

    history = []
    print("\n" + "=" * 65)
    print("FEDERATED HYBRID CNN-LSTM TRAINING SIMULATION")
    print(f"Rounds: {num_rounds} | Local Epochs: {local_epochs} | LR: {lr} | Clients: 3")
    print(f"Client Sample Distribution: H1={clients_data['hospital_1']['num_samples']}, "
          f"H2={clients_data['hospital_2']['num_samples']}, H3={clients_data['hospital_3']['num_samples']}")
    print("=" * 65)

    for rnd in range(1, num_rounds + 1):
        round_t0 = time.perf_counter()
        client_state_dicts = []
        client_sample_counts = []
        client_losses = []

        # Client local training
        for c_id in ["hospital_1", "hospital_2", "hospital_3"]:
            c_info = clients_data[c_id]
            # Copy global model
            local_model = HybridCNNLSTMModel(
                sequence_length=seq_len,
                num_features=num_feat,
                conv_filters=16,
                kernel_size=3,
                lstm_units=16
            )
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            # Train locally
            train_res = train_temporal_model(
                model=local_model,
                X_train=c_info["X"],
                y_train=c_info["y"],
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                random_seed=random_seed + rnd * 10,
                verbose=False
            )
            client_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            client_sample_counts.append(c_info["num_samples"])
            client_losses.append(train_res["final_train_loss"])

        # Central Aggregation
        agg_t0 = time.perf_counter()
        aggregated_weights = aggregate_state_dicts(client_state_dicts, client_sample_counts)
        global_model.load_state_dict(aggregated_weights)
        agg_elapsed = time.perf_counter() - agg_t0

        # Evaluate aggregated global model on holdout test set
        val_eval = evaluate_temporal_model(global_model, X_val, y_val)
        test_eval = evaluate_temporal_model(global_model, X_test, y_test)

        round_elapsed = time.perf_counter() - round_t0
        avg_client_loss = float(np.average(client_losses, weights=client_sample_counts))

        round_record = {
            "round": rnd,
            "average_client_loss": round(avg_client_loss, 4),
            "hospital_1_loss": round(client_losses[0], 4),
            "hospital_2_loss": round(client_losses[1], 4),
            "hospital_3_loss": round(client_losses[2], 4),
            "val_loss": val_eval["loss"],
            "val_roc_auc": val_eval["roc_auc"],
            "test_loss": test_eval["loss"],
            "test_accuracy": test_eval["accuracy"],
            "test_precision": test_eval["precision"],
            "test_recall": test_eval["recall"],
            "test_f1_score": test_eval["f1_score"],
            "test_roc_auc": test_eval["roc_auc"],
            "aggregation_time_sec": round(agg_elapsed, 5),
            "round_duration_sec": round(round_elapsed, 4)
        }
        history.append(round_record)

        print(f"Round {rnd:02d}/{num_rounds:02d} | Avg Loss: {avg_client_loss:.4f} | "
              f"Test AUC: {test_eval['roc_auc']:.4f} | Test F1: {test_eval['f1_score']:.4f} | "
              f"Test Acc: {test_eval['accuracy']:.4f} | Time: {round_elapsed:.3f}s")

    df_fed = pd.DataFrame(history)
    fed_csv_path = os.path.join(results_dir, "cnn_lstm_federated_history.csv")
    df_fed.to_csv(fed_csv_path, index=False)

    final_eval = evaluate_temporal_model(global_model, X_test, y_test)

    print("\nFederated CNN-LSTM Training Complete.")
    print(f"Final Global Test ROC-AUC: {final_eval['roc_auc']:.4f} | F1: {final_eval['f1_score']:.4f} | Accuracy: {final_eval['accuracy']:.4f}")
    print(f"History saved to: {fed_csv_path}")
    print("=" * 65)

    return {
        "global_model": global_model,
        "federated_history": df_fed,
        "final_test_metrics": final_eval,
        "num_rounds": num_rounds
    }


if __name__ == "__main__":
    # Quick self-test execution
    print("Running Baseline Comparison...")
    comp_res = run_cnn_lstm_baseline_comparison(epochs=15)
    print("\nRunning Federated CNN-LSTM Simulation...")
    fed_res = train_federated_cnn_lstm(num_rounds=5, local_epochs=3)

