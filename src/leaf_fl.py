"""
LEAF-FL: Lead-time and Equity-Aware Federated Learning for Healthcare Deterioration Prediction.

This module implements the LEAF-FL algorithm:
1. Sequential Causal CNN-LSTM model for continuous early warning across observation window (Months 1-6).
2. Clinical Warning Policy: Evaluates hazard/risk at each time step to determine earliest warning step t*.
3. Early-Warning & Equity Metrics:
   - Missed Event Rate (MER), False Alarm Rate (FAR), Late Warning Rate (LWR).
   - Average & Median Lead Time (months before future event in months 7-12).
   - Worst-Hospital Recall, Recall Dispersion Gap, and Standard Deviation.
4. Lead-time and Equity-Aware Aggregator:
   - Hospital warning-error score: E_k = a * MER_k + b * LWR_k + c * FAR_k
   - Aggregation weight: alpha_k = (1 - lambda) * (n_k / N) + lambda * softmax(tau * E_k)
   - Guarantees: When lambda = 0, exactly reproduces sample-weighted FedAvg.
"""

import os
import sys
import copy
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
    average_precision_score,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.leaf_data import (
    FEATURE_NAMES,
    OBSERVATION_WINDOW,
    FORECAST_HORIZON
)


# ==============================================================================
# 1. SEQUENTIAL CAUSAL CNN-LSTM MODEL
# ==============================================================================

class LEAFCausalCNNLSTM(nn.Module):
    """
    Causal 1D CNN + Unidirectional LSTM network for longitudinal patient forecasting.
    Inputs: (B, T=6, F=5) representing observations from Months 1 to 6.
    Outputs:
      - final_prob: (B, 1) risk probability at end of observation window.
      - seq_probs: (B, T, 1) sequential risk probability evaluated at each observation month.
    """
    def __init__(
        self,
        sequence_length: int = 6,
        num_features: int = 5,
        conv_filters: int = 16,
        kernel_size: int = 3,
        lstm_units: int = 16,
        dropout_rate: float = 0.15
    ):
        super(LEAFCausalCNNLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units

        # 1D CNN: causal padding so feature map at step t only depends on steps <= t
        # Left-padding with (kernel_size - 1)
        self.causal_pad = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=0
        )
        self.relu = nn.ReLU()

        # Unidirectional LSTM over time dimension
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classification Head
        self.fc1 = nn.Linear(lstm_units, 8)
        self.fc_out = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Parameters:
            x: Tensor of shape (B, T, F)
        Returns:
            final_prob: Tensor of shape (B, 1)
            seq_probs: Tensor of shape (B, T, 1)
        """
        B, T, F = x.shape
        # Permute for Conv1d: (B, F, T)
        x_in = x.permute(0, 2, 1)
        x_padded = self.causal_pad(x_in)
        x_conv = self.relu(self.conv1(x_padded))  # (B, conv_filters, T)

        # Permute back for LSTM: (B, T, conv_filters)
        x_lstm_in = x_conv.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm_in)        # (B, T, lstm_units)

        # Compute sequential hazard predictions for each step t in [1, T]
        # Applied across all time steps
        h_seq = self.relu(self.fc1(self.dropout(lstm_out)))
        seq_logits = self.fc_out(h_seq)
        seq_probs = self.sigmoid(seq_logits)      # (B, T, 1)

        final_prob = seq_probs[:, -1, :]          # (B, 1)
        return final_prob, seq_probs


# ==============================================================================
# 2. SEQUENTIAL WARNING POLICY & LEAD-TIME COMPUTATION
# ==============================================================================

def predict_sequential_warnings(
    model: nn.Module,
    X: np.ndarray,
    threshold: float = 0.50,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    Evaluate sequential warning policy across observation window (Months 1 to 6).

    Warning Policy:
    At each observation month m in {1, ..., 6}, if seq_prob[m] >= threshold,
    a clinical deterioration warning is triggered.
    The Warning Step t* is the earliest month where warning triggers.
    If no month exceeds threshold, warning_step = 0 (no warning).

    Returns:
        dict:
            - y_pred: binary prediction (B,) - 1 if any warning triggered, else 0
            - warning_step: integer array (B,) - month in 1..6 or 0
            - max_prob: maximum predicted probability across months 1..6 (B,)
            - final_prob: predicted probability at month 6 (B,)
            - seq_probs: complete sequential probabilities (B, 6)
    """
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        final_prob, seq_probs = model(x_t)

        final_p_np = final_prob.cpu().numpy().ravel()
        seq_p_np = seq_probs.cpu().numpy().squeeze(-1)  # (B, 6)

    B = X.shape[0]
    warning_step = np.zeros(B, dtype=np.int32)
    y_pred = np.zeros(B, dtype=np.int32)
    max_prob = np.max(seq_p_np, axis=1)

    for i in range(B):
        # Find earliest step where threshold is breached
        exceeded = np.where(seq_p_np[i] >= threshold)[0]
        if len(exceeded) > 0:
            warning_step[i] = int(exceeded[0] + 1)  # 1-indexed month (1 to 6)
            y_pred[i] = 1
        else:
            warning_step[i] = 0
            y_pred[i] = 0

    return {
        "y_pred": y_pred,
        "warning_step": warning_step,
        "max_prob": max_prob,
        "final_prob": final_p_np,
        "seq_probs": seq_p_np
    }


def compute_warning_metrics(
    y_true: np.ndarray,
    event_month: np.ndarray,
    y_pred: np.ndarray,
    warning_step: np.ndarray,
    y_prob: np.ndarray,
    min_lead_time: int = 2
) -> Dict[str, Any]:
    """
    Calculate comprehensive diagnostic, early-warning, and lead-time metrics.

    Parameters:
        y_true: ground truth binary deterioration (N,)
        event_month: ground truth event month (7-12) or 0 (N,)
        y_pred: binary warnings issued (N,)
        warning_step: earliest warning month (1-6) or 0 (N,)
        y_prob: predicted probabilities (N,)
        min_lead_time: minimum months required for a warning not to be 'late' (default: 2)

    Returns:
        dict containing standard diagnostic + early-warning metrics.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()
    ev_m = np.asarray(event_month, dtype=int).ravel()
    w_s = np.asarray(warning_step, dtype=int).ravel()
    prob = np.asarray(y_prob, dtype=float).ravel()

    N = len(y_t)
    P = int(np.sum(y_t == 1))
    N_neg = int(np.sum(y_t == 0))

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    spec = float(tn / max(1, tn + fp))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_t, prob)) if len(np.unique(y_t)) > 1 else 0.5
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, prob)) if len(np.unique(y_t)) > 1 else 0.0
    except Exception:
        pr_auc = 0.0

    # Early-Warning & Lead-Time Analysis
    # A true positive has y_t == 1 and w_s > 0
    tp_indices = np.where((y_t == 1) & (w_s > 0))[0]
    lead_times = []
    late_warning_count = 0

    for idx in tp_indices:
        t_event = ev_m[idx]  # in {7, ..., 12}
        t_warn = w_s[idx]    # in {1, ..., 6}
        lead = t_event - t_warn
        lead_times.append(lead)
        if lead < min_lead_time:
            late_warning_count += 1

    # Missed events: patients who actually deteriorated but no warning was triggered (fn)
    missed_event_rate = float(fn / max(1, P))

    # False alarm rate: patients who never deteriorated but received a warning (fp / N_neg)
    false_alarm_rate = float(fp / max(1, N_neg))

    # Late warning rate: warnings arriving too late (< min_lead_time) relative to actual event
    late_warning_rate = float(late_warning_count / max(1, P))

    # Lead time statistics (only over detected true events)
    if len(lead_times) > 0:
        avg_lead_time = float(np.mean(lead_times))
        median_lead_time = float(np.median(lead_times))
        pct_detected_before = float(len(lead_times) / max(1, P) * 100.0)
    else:
        avg_lead_time = 0.0
        median_lead_time = 0.0
        pct_detected_before = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "total_samples": N,
        "actual_positives": P,
        "actual_negatives": N_neg,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "missed_event_rate": missed_event_rate,
        "false_alarm_rate": false_alarm_rate,
        "late_warning_rate": late_warning_rate,
        "avg_lead_time_months": avg_lead_time,
        "median_lead_time_months": median_lead_time,
        "pct_detected_before_event": pct_detected_before,
        "lead_times_distribution": lead_times
    }


def compute_warning_error_score(
    metrics: Dict[str, Any],
    a: float = 0.50,
    b: float = 0.30,
    c: float = 0.20
) -> float:
    """
    Calculate composite hospital warning-error score:
        E_k = a * MissedEventRate_k + b * LateWarningRate_k + c * FalseAlarmRate_k

    Coefficients default to a=0.50, b=0.30, c=0.20 and are fully configurable.
    """
    if a < 0 or b < 0 or c < 0:
        raise ValueError("Coefficients a, b, c must be non-negative.")

    mer = metrics.get("missed_event_rate", 0.0)
    lwr = metrics.get("late_warning_rate", 0.0)
    far = metrics.get("false_alarm_rate", 0.0)

    # Normalize weights if sum != 1.0
    denom = a + b + c
    if denom <= 0:
        raise ValueError("Sum of coefficients a + b + c must be greater than zero.")
    a_norm = a / denom
    b_norm = b / denom
    c_norm = c / denom

    error_score = (a_norm * mer) + (b_norm * lwr) + (c_norm * far)
    return float(np.clip(error_score, 0.0, 1.0))


# ==============================================================================
# 3. LEAF-FL AGGREGATION WEIGHT CALCULATION
# ==============================================================================

def compute_leaf_aggregation_weights(
    client_sample_counts: List[int],
    client_error_scores: List[float],
    lambda_equity: float = 0.50,
    tau: float = 2.0
) -> List[float]:
    """
    Calculate Lead-time and Equity-Aware aggregation weights:
        alpha_k = (1 - lambda) * (n_k / N) + lambda * (exp(tau * E_k) / sum_j exp(tau * E_j))

    Properties:
    - lambda in [0, 1]: 0.0 reproduces standard sample-weighted FedAvg.
    - tau > 0: softmax temperature parameter.
    - Numerically stable softmax computation using max subtraction.
    - Output weights strictly sum to 1.0.

    Parameters:
        client_sample_counts: List of training sample sizes n_k per hospital.
        client_error_scores: List of warning-error scores E_k per hospital.
        lambda_equity: Equity weight parameter in [0.0, 1.0].
        tau: Temperature scaling parameter (tau > 0).

    Returns:
        List[float]: Normalized aggregation weights alpha_k.
    """
    if not (0.0 <= lambda_equity <= 1.0):
        raise ValueError(f"lambda_equity must be between 0.0 and 1.0, got {lambda_equity}.")
    if tau <= 0.0:
        raise ValueError(f"Temperature parameter tau must be strictly positive, got {tau}.")
    if len(client_sample_counts) == 0:
        raise ValueError("client_sample_counts cannot be empty.")
    if len(client_sample_counts) != len(client_error_scores):
        raise ValueError(
            f"Length mismatch: {len(client_sample_counts)} counts vs {len(client_error_scores)} error scores."
        )

    for n in client_sample_counts:
        if n <= 0:
            raise ValueError(f"Hospital sample count must be positive integer, got {n}.")

    total_samples = float(sum(client_sample_counts))
    sample_weights = np.array([float(n) / total_samples for n in client_sample_counts], dtype=float)

    # If lambda == 0.0, exact sample-weighted FedAvg
    if lambda_equity == 0.0:
        return [float(w) for w in sample_weights]

    # Numerically stable softmax: subtract max before exponentiation
    errors = np.array(client_error_scores, dtype=float)
    if np.any(np.isnan(errors)) or np.any(np.isinf(errors)):
        raise ValueError("Error scores contain NaN or infinite values.")

    scaled_errors = tau * errors
    max_scaled = np.max(scaled_errors)
    exp_errors = np.exp(scaled_errors - max_scaled)
    sum_exp = np.sum(exp_errors)

    if sum_exp <= 0.0 or np.isnan(sum_exp):
        equity_weights = np.ones(len(client_sample_counts)) / len(client_sample_counts)
    else:
        equity_weights = exp_errors / sum_exp

    # Linear combination
    combined = (1.0 - lambda_equity) * sample_weights + lambda_equity * equity_weights

    # Normalize to exact sum of 1.0
    final_weights = combined / np.sum(combined)
    return [float(w) for w in final_weights]


def leaf_aggregate_parameters(
    client_state_dicts: List[Dict[str, torch.Tensor]],
    aggregation_weights: List[float]
) -> Dict[str, torch.Tensor]:
    """
    Aggregate client neural network parameter state_dicts using LEAF-FL weights:
        W_global = sum_{k=1}^K alpha_k * W_k
    """
    if len(client_state_dicts) == 0:
        raise ValueError("client_state_dicts cannot be empty.")
    if len(client_state_dicts) != len(aggregation_weights):
        raise ValueError("Length mismatch between client state dicts and aggregation weights.")

    total_w = sum(aggregation_weights)
    if abs(total_w - 1.0) > 1e-5:
        norm_weights = [w / total_w for w in aggregation_weights]
    else:
        norm_weights = aggregation_weights

    aggregated_state = {}
    reference_dict = client_state_dicts[0]

    for key, ref_tensor in reference_dict.items():
        if ref_tensor.is_floating_point():
            accum = torch.zeros_like(ref_tensor)
            for k_idx, sd in enumerate(client_state_dicts):
                if key not in sd:
                    raise KeyError(f"Key '{key}' missing from client {k_idx} state_dict.")
                if sd[key].shape != ref_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for '{key}': expected {ref_tensor.shape}, got {sd[key].shape}"
                    )
                accum += norm_weights[k_idx] * sd[key]
            aggregated_state[key] = accum
        else:
            aggregated_state[key] = ref_tensor.clone()

    return aggregated_state


# ==============================================================================
# 4. LOCAL CLIENT TRAINING ENGINE
# ==============================================================================

def train_local_client(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.005,
    mu: float = 0.0,
    global_model_state: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, float]:
    """
    Train local client model on private observations using BCE loss with optional FedProx penalty.
    """
    model.train()
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    final_loss = 0.0

    for epoch in range(epochs):
        epoch_losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            final_prob, _ = model(x_batch)
            loss = criterion(final_prob, y_batch)

            # FedProx proximal term
            if mu > 0.0 and global_model_state is not None:
                prox_penalty = 0.0
                for name, param in model.named_parameters():
                    if name in global_model_state:
                        g_param = global_model_state[name].to(device)
                        prox_penalty += torch.sum((param - g_param) ** 2)
                loss = loss + (mu / 2.0) * prox_penalty

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        final_loss = float(np.mean(epoch_losses))

    return model, final_loss


# ==============================================================================
# 5. END-TO-END LEAF-FL EXPERIMENT RUNNER
# ==============================================================================

def run_leaf_fl_experiment(
    data_dict: Dict[str, Any],
    num_rounds: int = 5,
    local_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.005,
    lambda_equity: float = 0.50,
    tau: float = 2.0,
    coef_a: float = 0.50,
    coef_b: float = 0.30,
    coef_c: float = 0.20,
    mu: float = 0.0,
    random_seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Execute LEAF-FL training across the 3 heterogeneous hospital clients.

    Execution Lifecycle per Round:
    1. Distribute current global model to all hospitals.
    2. Each hospital performs local training on its private training observations.
    3. Coordinator evaluates local hospital models on each hospital's local validation set D_val,k.
       (STRICTLY NO TEST-SET EXPOSURE DURING TRAINING OR AGGREGATION).
    4. Compute warning-error scores E_k from validation MER_k, LWR_k, FAR_k.
    5. Compute LEAF-FL equity aggregation weights alpha_k.
    6. Aggregate local parameters: W_global = sum alpha_k * W_k.
    7. After final round: Evaluate global model on isolated Central Test Set and on each hospital test split.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    hospitals = data_dict["hospitals"]
    test_data = data_dict["test"]

    # Initialize Global Model
    global_model = LEAFCausalCNNLSTM(
        sequence_length=OBSERVATION_WINDOW,
        num_features=len(FEATURE_NAMES),
        conv_filters=16,
        kernel_size=3,
        lstm_units=16
    ).to(device)

    round_history = []
    weight_history = []

    print("\n" + "=" * 68)
    print("LEAF-FL EXPERIMENT RUNNER (LEAD-TIME & EQUITY-AWARE FEDERATED LEARNING)")
    print(f"Rounds: {num_rounds} | Local Epochs: {local_epochs} | LR: {lr} | Lambda: {lambda_equity} | Tau: {tau}")
    print(f"Error Coefficients: a(MER)={coef_a}, b(LWR)={coef_b}, c(FAR)={coef_c}")
    print("=" * 68)

    t0_exp = time.perf_counter()

    for rnd in range(1, num_rounds + 1):
        round_t0 = time.perf_counter()
        current_global_state = copy.deepcopy(global_model.state_dict())

        client_state_dicts = []
        client_sample_counts = []
        client_error_scores = []
        client_val_metrics = {}
        client_train_losses = {}

        # 1. Parallel Local Training & Local Validation Performance Assessment
        for h_id, h_info in hospitals.items():
            tr_data = h_info["train"]
            val_data = h_info["val"]
            n_samples = tr_data["num_samples"]

            # Initialize local model with global parameters
            local_model = LEAFCausalCNNLSTM(
                sequence_length=OBSERVATION_WINDOW,
                num_features=len(FEATURE_NAMES),
                conv_filters=16,
                kernel_size=3,
                lstm_units=16
            ).to(device)
            local_model.load_state_dict(copy.deepcopy(current_global_state))

            # Train on local training split
            local_model, train_loss = train_local_client(
                model=local_model,
                X_train=tr_data["X"],
                y_train=tr_data["y"],
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                mu=mu,
                global_model_state=current_global_state,
                device=device
            )
            client_train_losses[h_id] = train_loss
            client_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            client_sample_counts.append(n_samples)

            # Evaluate on Local Validation Split (NO TEST DATA)
            val_preds = predict_sequential_warnings(local_model, val_data["X"], threshold=0.50, device=device)
            val_met = compute_warning_metrics(
                y_true=val_data["y"],
                event_month=val_data["event_month"],
                y_pred=val_preds["y_pred"],
                warning_step=val_preds["warning_step"],
                y_prob=val_preds["max_prob"]
            )
            client_val_metrics[h_id] = val_met

            # Calculate warning-error score E_k
            e_k = compute_warning_error_score(val_met, a=coef_a, b=coef_b, c=coef_c)
            client_error_scores.append(e_k)

        # 2. Compute LEAF-FL Aggregation Weights
        alpha_weights = compute_leaf_aggregation_weights(
            client_sample_counts=client_sample_counts,
            client_error_scores=client_error_scores,
            lambda_equity=lambda_equity,
            tau=tau
        )

        # 3. Aggregate Model Parameters
        t0_agg = time.perf_counter()
        aggregated_state = leaf_aggregate_parameters(client_state_dicts, alpha_weights)
        agg_duration = time.perf_counter() - t0_agg

        # Update Global Model
        global_model.load_state_dict(aggregated_state)

        round_duration = time.perf_counter() - round_t0

        # Log round telemetry
        h_ids = list(hospitals.keys())
        round_log = {
            "round": rnd,
            "lambda_equity": lambda_equity,
            "tau": tau,
            "hospital_1_loss": client_train_losses[h_ids[0]],
            "hospital_2_loss": client_train_losses[h_ids[1]],
            "hospital_3_loss": client_train_losses[h_ids[2]],
            "average_train_loss": float(np.mean(list(client_train_losses.values()))),
            "hospital_1_error_score": client_error_scores[0],
            "hospital_2_error_score": client_error_scores[1],
            "hospital_3_error_score": client_error_scores[2],
            "hospital_1_weight": alpha_weights[0],
            "hospital_2_weight": alpha_weights[1],
            "hospital_3_weight": alpha_weights[2],
            "aggregation_time_seconds": round(agg_duration, 6),
            "round_duration_seconds": round(round_duration, 4)
        }
        round_history.append(round_log)

        weight_history.append({
            "round": rnd,
            "Hospital 1 (Specialty)": alpha_weights[0],
            "Hospital 2 (Community)": alpha_weights[1],
            "Hospital 3 (Rural)": alpha_weights[2]
        })

        print(f"Round {rnd}/{num_rounds} [{round_duration:.2f}s] - "
              f"Loss: {round_log['average_train_loss']:.4f} | "
              f"Error Scores: H1={client_error_scores[0]:.3f}, H2={client_error_scores[1]:.3f}, H3={client_error_scores[2]:.3f} | "
              f"Weights: H1={alpha_weights[0]:.3f}, H2={alpha_weights[1]:.3f}, H3={alpha_weights[2]:.3f}")

    total_experiment_time = time.perf_counter() - t0_exp

    # ==========================================================================
    # FINAL EVALUATION: UNSEEN CENTRAL TEST SET & HOSPITAL PERFORMANCE
    # ==========================================================================
    test_preds = predict_sequential_warnings(global_model, test_data["X"], threshold=0.50, device=device)
    global_test_metrics = compute_warning_metrics(
        y_true=test_data["y"],
        event_month=test_data["event_month"],
        y_pred=test_preds["y_pred"],
        warning_step=test_preds["warning_step"],
        y_prob=test_preds["max_prob"]
    )

    # Evaluate Global Model on each hospital's local validation split to assess equity
    hospital_evaluations = {}
    hospital_recalls = []
    hospital_aurocs = []

    for h_id, h_info in hospitals.items():
        v_data = h_info["val"]
        h_preds = predict_sequential_warnings(global_model, v_data["X"], threshold=0.50, device=device)
        h_met = compute_warning_metrics(
            y_true=v_data["y"],
            event_month=v_data["event_month"],
            y_pred=h_preds["y_pred"],
            warning_step=h_preds["warning_step"],
            y_prob=h_preds["max_prob"]
        )
        hospital_evaluations[h_id] = {
            "name": h_info["name"],
            "metrics": h_met
        }
        hospital_recalls.append(h_met["recall"])
        hospital_aurocs.append(h_met["roc_auc"])

    # Equity Metrics across participating hospitals
    worst_hospital_recall = float(np.min(hospital_recalls))
    worst_hospital_auroc = float(np.min(hospital_aurocs))
    max_recall_gap = float(np.max(hospital_recalls) - np.min(hospital_recalls))
    std_hospital_recall = float(np.std(hospital_recalls))

    equity_metrics = {
        "worst_hospital_recall": worst_hospital_recall,
        "worst_hospital_auroc": worst_hospital_auroc,
        "max_hospital_recall_gap": max_recall_gap,
        "std_hospital_recall": std_hospital_recall,
        "hospital_recalls": {h_id: hospital_evaluations[h_id]["metrics"]["recall"] for h_id in hospitals}
    }

    print("\n" + "=" * 68)
    print("FINAL EVALUATION ON CENTRAL TEST SET (150 PATIENTS)")
    print(f"Accuracy:  {global_test_metrics['accuracy']:.4f}")
    print(f"Recall:    {global_test_metrics['recall']:.4f} (Sensitivity)")
    print(f"Precision: {global_test_metrics['precision']:.4f}")
    print(f"F1-Score:  {global_test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {global_test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {global_test_metrics['pr_auc']:.4f}")
    print(f"Missed Event Rate:        {global_test_metrics['missed_event_rate']:.2%}")
    print(f"False Alarm Rate:         {global_test_metrics['false_alarm_rate']:.2%}")
    print(f"Average Warning Lead Time:{global_test_metrics['avg_lead_time_months']:.2f} months")
    print(f"Equity - Worst Recall:    {worst_hospital_recall:.4f}")
    print(f"Equity - Recall Gap:      {max_recall_gap:.4f}")
    print("=" * 68)

    return {
        "model": global_model,
        "global_test_metrics": global_test_metrics,
        "hospital_evaluations": hospital_evaluations,
        "equity_metrics": equity_metrics,
        "round_history": pd.DataFrame(round_history),
        "weight_history": pd.DataFrame(weight_history),
        "experiment_duration_seconds": total_experiment_time,
        "config": {
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "lambda_equity": lambda_equity,
            "tau": tau,
            "a": coef_a,
            "b": coef_b,
            "c": coef_c,
            "mu": mu,
            "random_seed": random_seed
        }
    }


if __name__ == "__main__":
    from src.leaf_data import prepare_and_save_leaf_data
    print("Testing LEAF-FL experiment pipeline...")
    data = prepare_and_save_leaf_data()
    res = run_leaf_fl_experiment(data, num_rounds=3, local_epochs=3, lambda_equity=0.5, tau=2.0)
    print("LEAF-FL test run completed successfully.")

