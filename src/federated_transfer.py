"""
Federated Transfer Learning (FTL) Module for Healthcare Analytics Simulation.

This module implements a simulated Federated Transfer Learning workflow across two institutions:
1. Source Hospital (Large Metropolitan Medical Center):
   - Large synthetic patient dataset (1,000 training, 200 test records).
   - General adult demographic distribution (mean age 50, standard vital ranges).
   - Pre-trains a source predictive model on healthcare risk classification.
2. Target Hospital (Community / Geriatric Care Clinic):
   - Small, resource-constrained synthetic dataset (200 training pool, 100 isolated test records).
   - Domain Shift (Covariate Shift): Older demographic (mean age 68), elevated blood pressure (mean 142 mmHg),
     higher glycemic variance, altered risk profile.
   - Limited labeled records: Supports training with target data fractions (10%, 25%, 50%, 100%).
   - Objective: Transfer learned representation capabilities from Source to boost Target performance.
3. Transfer Strategies:
   - Full Fine-Tuning: Initialize target model with source parameters and fine-tune all layers.
   - Feature Extraction: Freeze transferred feature extractor (Layer 1: W1, b1) and train only the prediction head (Layer 2: W2, b2).
4. Baselines for Rigorous Comparison:
   - Target-Only (Scratch): Fresh initialization trained solely on target training data.
   - Source-Only (Zero-Shot): Source model evaluated directly on target test set without adaptation.
   - Centralized Combined: Non-federated reference upper-bound trained on combined source + target data.
"""

import os
import sys
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
    confusion_matrix
)

# Ensure project root is accessible on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import HealthcareMLP
from src.data_preparation import FEATURE_COLUMNS, TARGET_COLUMN

# Domain configurations
SOURCE_DOMAIN_CONFIG = {
    "name": "Metropolitan Medical Center (Source Hospital)",
    "description": "Large urban medical center with diverse adult patient population.",
    "sample_size": 1200,
    "train_samples": 1000,
    "test_samples": 200,
    "age_mean": 50.0,
    "age_std": 12.0,
    "bp_mean": 124.0,
    "bp_std": 15.0,
    "glucose_mean": 108.0,
    "glucose_std": 25.0,
    "bmi_mean": 26.8,
    "bmi_std": 4.8,
    "chol_mean": 198.0,
    "chol_std": 32.0,
    "noise_std": 0.45
}

TARGET_DOMAIN_CONFIG = {
    "name": "Geriatric Community Clinic (Target Hospital)",
    "description": "Specialized senior health center with older demographic and higher cardiovascular/metabolic risks.",
    "sample_size": 300,
    "train_samples": 200,
    "test_samples": 100,
    "age_mean": 68.0,
    "age_std": 8.0,
    "bp_mean": 142.0,
    "bp_std": 18.0,
    "glucose_mean": 130.0,
    "glucose_std": 35.0,
    "bmi_mean": 29.2,
    "bmi_std": 5.5,
    "chol_mean": 225.0,
    "chol_std": 40.0,
    "noise_std": 0.50
}


# ==============================================================================
# DATASET GENERATION & PARTITIONING FOR TRANSFER LEARNING
# ==============================================================================

def generate_domain_dataset(
    config: Dict[str, Any],
    prefix: str = "SRC",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic clinical records tailored to a specific domain configuration.
    Assigns unique deterministic patient IDs with the specified prefix.
    """
    rng = np.random.RandomState(random_seed)
    n = config["sample_size"]

    age = rng.normal(loc=config["age_mean"], scale=config["age_std"], size=n)
    age = np.clip(age, 20.0, 90.0).round().astype(int)

    bp = rng.normal(loc=config["bp_mean"], scale=config["bp_std"], size=n)
    bp = np.clip(bp, 85.0, 200.0).round(1)

    glucose = rng.normal(loc=config["glucose_mean"], scale=config["glucose_std"], size=n)
    glucose = np.clip(glucose, 65.0, 280.0).round(1)

    bmi = rng.normal(loc=config["bmi_mean"], scale=config["bmi_std"], size=n)
    bmi = np.clip(bmi, 17.0, 50.0).round(1)

    chol = rng.normal(loc=config["chol_mean"], scale=config["chol_std"], size=n)
    chol = np.clip(chol, 120.0, 350.0).round(1)

    # Standardized risk score calculation with domain shift
    z_age = (age - config["age_mean"]) / max(1.0, config["age_std"])
    z_bp = (bp - config["bp_mean"]) / max(1.0, config["bp_std"])
    z_glu = (glucose - config["glucose_mean"]) / max(1.0, config["glucose_std"])
    z_bmi = (bmi - config["bmi_mean"]) / max(1.0, config["bmi_std"])
    z_chol = (chol - config["chol_mean"]) / max(1.0, config["chol_std"])

    noise = rng.normal(loc=0.0, scale=config["noise_std"], size=n)
    risk_score = (
        0.50 * z_age +
        0.55 * z_bp +
        0.65 * z_glu +
        0.45 * z_bmi +
        0.40 * z_chol +
        noise
    )

    prob = 1.0 / (1.0 + np.exp(-risk_score))
    risk_label = (prob >= 0.50).astype(int)

    pids = [f"{prefix}_{i+1:04d}" for i in range(n)]

    df = pd.DataFrame({
        "patient_id": pids,
        "age": age,
        "blood_pressure": bp,
        "glucose_level": glucose,
        "bmi": bmi,
        "cholesterol_level": chol,
        "risk_label": risk_label
    })

    return df


def prepare_ftl_datasets(
    output_dir: str = os.path.join("data", "ftl"),
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Generate and persist partitioned source and target datasets:
    - source_train.csv (1,000 records)
    - source_test.csv (200 records)
    - target_train_pool.csv (200 records)
    - target_test.csv (100 records)
    - ftl_scaler.json (Source domain normalization parameters)

    Returns:
        Dict[str, str]: Dictionary mapping partition names to saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Source domain dataset
    df_src = generate_domain_dataset(SOURCE_DOMAIN_CONFIG, prefix="SRC", random_seed=random_seed)
    n_src_train = SOURCE_DOMAIN_CONFIG["train_samples"]
    src_train = df_src.iloc[:n_src_train].copy()
    src_test = df_src.iloc[n_src_train:].copy()

    # 2. Generate Target domain dataset
    df_tgt = generate_domain_dataset(TARGET_DOMAIN_CONFIG, prefix="TGT", random_seed=random_seed + 100)
    n_tgt_train = TARGET_DOMAIN_CONFIG["train_samples"]
    tgt_train = df_tgt.iloc[:n_tgt_train].copy()
    tgt_test = df_tgt.iloc[n_tgt_train:].copy()

    # 3. Compute Source Scaler (exclusively on source training set)
    scaler = {
        "mean": {col: float(src_train[col].mean()) for col in FEATURE_COLUMNS},
        "std": {col: float(max(src_train[col].std(), 1e-7)) for col in FEATURE_COLUMNS}
    }

    # 4. Save CSV files
    paths = {
        "source_train": os.path.join(output_dir, "source_train.csv"),
        "source_test": os.path.join(output_dir, "source_test.csv"),
        "target_train_pool": os.path.join(output_dir, "target_train_pool.csv"),
        "target_test": os.path.join(output_dir, "target_test.csv"),
        "scaler": os.path.join(output_dir, "ftl_scaler.json")
    }

    src_train.to_csv(paths["source_train"], index=False)
    src_test.to_csv(paths["source_test"], index=False)
    tgt_train.to_csv(paths["target_train_pool"], index=False)
    tgt_test.to_csv(paths["target_test"], index=False)

    with open(paths["scaler"], "w") as f:
        json.dump(scaler, f, indent=4)

    return paths


def apply_domain_scaler(df: pd.DataFrame, scaler: Dict[str, Any]) -> np.ndarray:
    """
    Normalize dataframe features using pre-computed scaling parameters.
    Returns normalized feature matrix of shape (N, 5).
    """
    X = df[FEATURE_COLUMNS].values.astype(float).copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        m = scaler["mean"].get(col, 0.0)
        s = scaler["std"].get(col, 1.0)
        s = s if s > 1e-12 else 1.0
        X[:, i] = (X[:, i] - m) / s
    return X


# ==============================================================================
# TRANSFER LEARNING MODEL IMPLEMENTATION
# ==============================================================================

class TransferHealthcareModel:
    """
    Two-layer Neural Network supporting transfer learning strategies:
    - Feature Extractor: W1 (5 -> 8), b1 (1 -> 8) [48 parameters]
    - Prediction Head: W2 (8 -> 1), b2 (1 -> 1)   [9 parameters]
    - Total Parameters: 57 parameters
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 8,
        output_dim: int = 1,
        random_seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.random_seed = random_seed
        self.mlp = HealthcareMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            random_seed=random_seed
        )

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return deep copy of current parameters."""
        return self.mlp.get_parameters()

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Load external parameters into model."""
        self.mlp.set_parameters(params)

    def load_feature_extractor(self, source_params: Dict[str, np.ndarray]) -> None:
        """Load only Layer 1 parameters (W1, b1) from source parameters."""
        current = self.mlp.get_parameters()
        current["W1"] = np.asarray(source_params["W1"], dtype=float).copy()
        current["b1"] = np.asarray(source_params["b1"], dtype=float).copy()
        self.mlp.set_parameters(current)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.02,
        transfer_strategy: str = "full_fine_tuning",
        verbose: bool = False
    ) -> List[float]:
        """
        Train or fine-tune model with the specified transfer strategy:
        - 'full_fine_tuning': Updates both feature extractor (W1, b1) and prediction head (W2, b2).
        - 'feature_extraction': Freezes feature extractor (W1, b1); updates only prediction head (W2, b2).
        - 'scratch': Fresh unconstrained training on all layers.

        Returns:
            List[float]: Per-epoch loss history.
        """
        valid_strategies = ["full_fine_tuning", "feature_extraction", "scratch"]
        if transfer_strategy not in valid_strategies:
            raise ValueError(f"Unknown transfer strategy '{transfer_strategy}'. Valid: {valid_strategies}")

        N = X_train.shape[0]
        y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
        loss_history = []
        rng = np.random.RandomState(self.random_seed)

        for ep in range(epochs):
            indices = np.arange(N)
            rng.shuffle(indices)
            epoch_losses = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                b_idx = indices[start:end]
                X_b = X_train[b_idx]
                y_b = y_train[b_idx]

                # Forward pass
                A2, cache = self.mlp.forward(X_b)
                loss_val = self.mlp.compute_loss(y_b, A2)
                epoch_losses.append(loss_val)

                # Backward pass
                grads = self.mlp.backward(cache, y_b)

                # Apply transfer strategy parameter updates
                if transfer_strategy == "feature_extraction":
                    # Freeze feature extractor: only update W2, b2
                    self.mlp.parameters["W2"] -= learning_rate * grads["W2"]
                    self.mlp.parameters["b2"] -= learning_rate * grads["b2"]
                else:
                    # Update all parameters (full fine-tuning or scratch)
                    self.mlp.update_parameters(grads, learning_rate=learning_rate)

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            loss_history.append(mean_loss)
            if verbose and (ep + 1) % max(1, epochs // 5) == 0:
                print(f"Epoch {ep+1}/{epochs} - Loss: {mean_loss:.4f} ({transfer_strategy})")

        return loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted risk probabilities."""
        return self.mlp.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """Compute binary predictions."""
        return self.mlp.predict(X, threshold=threshold)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate classification metrics on a test set.
        Returns accuracy, precision, recall, f1_score, roc_auc, confusion_matrix.
        """
        y_true = np.asarray(y_test, dtype=int).ravel()
        probs = self.predict_proba(X_test).ravel()
        preds = (probs >= 0.50).astype(int)

        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, probs))
        except Exception:
            auc = None

        cm = confusion_matrix(y_true, preds).tolist()

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "test_samples": len(y_true)
        }


# ==============================================================================
# FTL EXPERIMENT WORKFLOW & BENCHMARKING
# ==============================================================================

def train_source_model(
    source_train_df: pd.DataFrame,
    scaler: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.02,
    random_seed: int = 42
) -> Tuple[TransferHealthcareModel, List[float], float]:
    """
    Train source domain predictive model on the full source training dataset.
    Returns: (trained_model, loss_history, training_latency_seconds)
    """
    X_src = apply_domain_scaler(source_train_df, scaler)
    y_src = source_train_df[TARGET_COLUMN].values

    model = TransferHealthcareModel(random_seed=random_seed)
    t0 = time.perf_counter()
    history = model.fit(
        X_src, y_src,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        transfer_strategy="scratch",
        verbose=False
    )
    duration = time.perf_counter() - t0

    return model, history, duration


def run_ftl_experiment(
    fractions: Optional[List[float]] = None,
    source_epochs: int = 40,
    target_epochs: int = 30,
    learning_rate: float = 0.02,
    batch_size: int = 32,
    random_seed: int = 42,
    data_dir: str = os.path.join("data", "ftl"),
    output_dir: str = os.path.join("results", "metrics")
) -> Dict[str, Any]:
    """
    Execute full Federated Transfer Learning experimental workflow across multiple
    target labeled-data fractions:
    - Target-Only (Trained from scratch on target subset)
    - Source-Only (Direct zero-shot evaluation of source model on target test set)
    - FTL Full Fine-Tuning (Source initialized, all layers fine-tuned)
    - FTL Feature Extraction (Source feature extractor frozen, head trained)
    - Centralized Combined Reference (Trained on combined source + target subset)

    Parameters:
        fractions (List[float]): Labeled target data proportions to evaluate (e.g. [0.1, 0.25, 0.5, 1.0]).
        source_epochs (int): Pre-training epochs on Source Hospital dataset.
        target_epochs (int): Fine-tuning epochs on Target Hospital subsets.
        learning_rate (float): Optimization step size.
        batch_size (int): Mini-batch size.
        random_seed (int): Deterministic initialization seed.
        data_dir (str): Storage path for FTL datasets.
        output_dir (str): Destination for experimental metrics.

    Returns:
        dict: Complete FTL experiment results and comparative summaries.
    """
    if fractions is None:
        fractions = [0.10, 0.25, 0.50, 1.00]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ensure FTL datasets exist
    data_paths = {
        "source_train": os.path.join(data_dir, "source_train.csv"),
        "source_test": os.path.join(data_dir, "source_test.csv"),
        "target_train_pool": os.path.join(data_dir, "target_train_pool.csv"),
        "target_test": os.path.join(data_dir, "target_test.csv"),
        "scaler": os.path.join(data_dir, "ftl_scaler.json")
    }
    if not all(os.path.exists(p) for p in data_paths.values()):
        data_paths = prepare_ftl_datasets(output_dir=data_dir, random_seed=random_seed)

    df_src_train = pd.read_csv(data_paths["source_train"])
    df_tgt_pool = pd.read_csv(data_paths["target_train_pool"])
    df_tgt_test = pd.read_csv(data_paths["target_test"])

    with open(data_paths["scaler"], "r") as f:
        scaler = json.load(f)

    # Prepare normalized target test data (isolated central test set)
    X_tgt_test = apply_domain_scaler(df_tgt_test, scaler)
    y_tgt_test = df_tgt_test[TARGET_COLUMN].values

    # 2. Train Source Domain Model
    print("=" * 70)
    print("PHASE 13: FEDERATED TRANSFER LEARNING (FTL)")
    print(f"Source: {len(df_src_train)} records | Target Pool: {len(df_tgt_pool)} records | Target Test: {len(df_tgt_test)} records")
    print("=" * 70)
    print("Step 1: Training Source Model on Metropolitan Medical Center data...")

    source_model, src_history, src_time = train_source_model(
        source_train_df=df_src_train,
        scaler=scaler,
        epochs=source_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed
    )
    source_parameters = source_model.get_parameters()

    # Zero-shot evaluation of Source-only model on target test set
    source_only_eval = source_model.evaluate(X_tgt_test, y_tgt_test)
    print(f"Source-Only Model (Zero-Shot on Target Test): Accuracy = {source_only_eval['accuracy']:.2%}, F1 = {source_only_eval['f1_score']:.4f}")

    # 3. Iterate over Target Labeled-Data Fractions
    results_by_fraction = {}
    comparison_table_rows = []

    for frac in fractions:
        n_samples = max(10, int(round(len(df_tgt_pool) * frac)))
        df_tgt_subset = df_tgt_pool.iloc[:n_samples].copy()
        X_tgt_sub = apply_domain_scaler(df_tgt_subset, scaler)
        y_tgt_sub = df_tgt_subset[TARGET_COLUMN].values

        print(f"\nEvaluating Target Fraction: {frac:.0%} ({n_samples} labeled patient records)...")

        # Approach A: Target-Only (Trained from scratch)
        m_target = TransferHealthcareModel(random_seed=random_seed + int(frac * 100))
        t0 = time.perf_counter()
        h_target = m_target.fit(
            X_tgt_sub, y_tgt_sub,
            epochs=target_epochs,
            batch_size=min(batch_size, n_samples),
            learning_rate=learning_rate,
            transfer_strategy="scratch",
            verbose=False
        )
        t_target = time.perf_counter() - t0
        eval_target = m_target.evaluate(X_tgt_test, y_tgt_test)

        # Approach B: FTL Feature Extraction (Layer 1 frozen, Layer 2 trained)
        m_fe = TransferHealthcareModel(random_seed=random_seed + int(frac * 100))
        m_fe.set_parameters(source_parameters)  # Initialize from source
        t0 = time.perf_counter()
        h_fe = m_fe.fit(
            X_tgt_sub, y_tgt_sub,
            epochs=target_epochs,
            batch_size=min(batch_size, n_samples),
            learning_rate=learning_rate,
            transfer_strategy="feature_extraction",
            verbose=False
        )
        t_fe = time.perf_counter() - t0
        eval_fe = m_fe.evaluate(X_tgt_test, y_tgt_test)

        # Approach C: FTL Full Fine-Tuning (All layers fine-tuned)
        m_fft = TransferHealthcareModel(random_seed=random_seed + int(frac * 100))
        m_fft.set_parameters(source_parameters)  # Initialize from source
        t0 = time.perf_counter()
        h_fft = m_fft.fit(
            X_tgt_sub, y_tgt_sub,
            epochs=target_epochs,
            batch_size=min(batch_size, n_samples),
            learning_rate=learning_rate,
            transfer_strategy="full_fine_tuning",
            verbose=False
        )
        t_fft = time.perf_counter() - t0
        eval_fft = m_fft.evaluate(X_tgt_test, y_tgt_test)

        # Approach D: Centralized Combined (Non-federated reference baseline)
        df_combined = pd.concat([df_src_train, df_tgt_subset], ignore_index=True)
        X_comb = apply_domain_scaler(df_combined, scaler)
        y_comb = df_combined[TARGET_COLUMN].values
        m_comb = TransferHealthcareModel(random_seed=random_seed + int(frac * 100))
        t0 = time.perf_counter()
        h_comb = m_comb.fit(
            X_comb, y_comb,
            epochs=target_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            transfer_strategy="scratch",
            verbose=False
        )
        t_comb = time.perf_counter() - t0
        eval_comb = m_comb.evaluate(X_tgt_test, y_tgt_test)

        frac_key = f"{int(frac*100)}%"
        results_by_fraction[frac_key] = {
            "fraction": frac,
            "target_samples": n_samples,
            "target_only": {"metrics": eval_target, "training_time": round(t_target, 4)},
            "source_only": {"metrics": source_only_eval, "training_time": 0.0},
            "ftl_feature_extraction": {"metrics": eval_fe, "training_time": round(t_fe, 4)},
            "ftl_full_fine_tuning": {"metrics": eval_fft, "training_time": round(t_fft, 4)},
            "centralized_combined": {"metrics": eval_comb, "training_time": round(t_comb, 4)}
        }

        # Tabular comparison rows
        for approach_name, model_type, ev, t_sec in [
            ("Target-Only (Scratch)", "Local Baseline", eval_target, t_target),
            ("Source-Only (Zero-Shot)", "Direct Transfer (No adaptation)", source_only_eval, 0.0),
            ("FTL: Feature Extraction", "Frozen W1, b1 (Trained Head)", eval_fe, t_fe),
            ("FTL: Full Fine-Tuning", "Source Initialized (All layers tuned)", eval_fft, t_fft),
            ("Centralized Combined (Reference)", "Non-Federated Upper-Bound", eval_comb, t_comb)
        ]:
            comparison_table_rows.append({
                "Target Labeled Fraction": f"{int(frac*100)}%",
                "Target Sample Count": n_samples,
                "Approach": approach_name,
                "Strategy / Model Type": model_type,
                "Accuracy": round(ev["accuracy"], 4),
                "Precision": round(ev["precision"], 4),
                "Recall": round(ev["recall"], 4),
                "F1-Score": round(ev["f1_score"], 4),
                "ROC-AUC": round(ev["roc_auc"], 4) if ev.get("roc_auc") is not None else None,
                "Training Time (s)": round(t_sec, 4)
            })

    # 4. Save results to CSV and JSON
    df_comp = pd.DataFrame(comparison_table_rows)
    csv_path = os.path.join(output_dir, "ftl_fraction_comparison.csv")
    df_comp.to_csv(csv_path, index=False)

    # 5. Communication analysis for FTL
    from src.communication_analysis import calculate_ftl_communication_volume
    comm_summary = calculate_ftl_communication_volume()

    summary_payload = {
        "algorithm": "Federated Transfer Learning",
        "paradigm": "FTL (Cross-Domain Representation Transfer)",
        "source_domain": SOURCE_DOMAIN_CONFIG,
        "target_domain": TARGET_DOMAIN_CONFIG,
        "experimental_setup": {
            "source_training_samples": len(df_src_train),
            "target_training_pool_samples": len(df_tgt_pool),
            "target_test_samples": len(df_tgt_test),
            "evaluated_fractions": fractions,
            "source_epochs": source_epochs,
            "target_epochs": target_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "random_seed": random_seed
        },
        "source_pretraining": {
            "source_training_time_seconds": round(src_time, 4),
            "final_source_loss": round(src_history[-1], 6)
        },
        "results_by_fraction": results_by_fraction,
        "communication_analysis": comm_summary,
        "comparison_table": comparison_table_rows
    }

    json_path = os.path.join(output_dir, "ftl_evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_payload, f, indent=4)

    # Training history of 100% full fine-tuning for dashboard convergence plotting
    hist_records = [
        {"epoch": ep + 1, "source_pretrain_loss": round(src_l, 6)}
        for ep, src_l in enumerate(src_history)
    ]
    pd.DataFrame(hist_records).to_csv(os.path.join(output_dir, "ftl_training_history.csv"), index=False)

    print("\n" + "=" * 70)
    print("FTL EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    # Print summary table at 25% and 100% fractions
    print(df_comp[df_comp["Target Labeled Fraction"].isin(["25%", "100%"])][["Target Labeled Fraction", "Approach", "Accuracy", "F1-Score", "ROC-AUC"]].to_string(index=False))
    print("=" * 70)

    return summary_payload


if __name__ == "__main__":
    run_ftl_experiment()

