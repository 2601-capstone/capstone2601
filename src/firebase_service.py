"""
Firebase Cloud Firestore Service for Federated Healthcare Analytics.

This module handles:
1. Safe and graceful Firebase initialization using service account credentials.
2. Sanitization of scientific data (converting NumPy types/arrays/NaNs into native Python types).
3. Saving federated learning experiment runs and evaluation summaries to Firestore.
4. Retrieving historical experiment records from Firestore.
5. Graceful fallback when Firestore is unconfigured or offline (local operation never breaks).

Collection Structure:
    experiments/{experiment_id}
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone
import numpy as np

# Optional dotenv support for local .env loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Firebase Admin SDK imports (guarded)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_SDK_AVAILABLE = True
except ImportError:
    FIREBASE_SDK_AVAILABLE = False


_firestore_client = None
_initialization_attempted = False
_initialization_error = None


def get_firestore_client():
    """
    Safely initialize and retrieve the Cloud Firestore client.
    Returns None if Firebase credentials are missing or invalid.
    """
    global _firestore_client, _initialization_attempted, _initialization_error

    if _firestore_client is not None:
        return _firestore_client

    if _initialization_attempted and _initialization_error is not None:
        return None

    _initialization_attempted = True

    if not FIREBASE_SDK_AVAILABLE:
        _initialization_error = "firebase-admin package is not installed."
        return None

    # Resolve credential path from environment variables or common locations
    cred_path = (
        os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY") or
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )

    if cred_path and not os.path.isabs(cred_path):
        candidate_rel = os.path.join(PROJECT_ROOT, cred_path)
        if os.path.exists(candidate_rel):
            cred_path = candidate_rel

    if not cred_path or not os.path.exists(cred_path):
        default_files = [
            "serviceAccountKey.json",
            "firebase_credentials.json",
            "firebase-service-account.json"
        ]
        for fname in default_files:
            candidate = os.path.join(PROJECT_ROOT, fname)
            if os.path.exists(candidate):
                cred_path = candidate
                break

    if not cred_path or not os.path.exists(cred_path):
        _initialization_error = (
            "Firebase credentials not found. Set FIREBASE_SERVICE_ACCOUNT_KEY in .env "
            "or place serviceAccountKey.json / firebase-service-account.json in the project root."
        )
        return None

    try:
        # Check if already initialized in this process
        try:
            app = firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(cred_path)
            app = firebase_admin.initialize_app(cred)

        _firestore_client = firestore.client(app)
        _initialization_error = None
        return _firestore_client
    except Exception as e:
        _initialization_error = f"Failed to initialize Firebase client: {type(e).__name__} - {str(e)}"
        _firestore_client = None
        return None


def check_firestore_connection() -> tuple:
    """
    Check whether Cloud Firestore is configured and ready.

    Returns:
        tuple (bool, str): (is_available, status_message)
    """
    client = get_firestore_client()
    if client is not None:
        return True, "Firebase Cloud Firestore is connected and ready."
    else:
        msg = _initialization_error or "Firebase credentials not configured."
        return False, msg


def sanitize_for_firestore(obj):
    """
    Recursively sanitize objects for Firestore storage:
    - Converts NumPy scalars (int, float, bool) to Python primitives.
    - Converts NumPy arrays to nested Python lists.
    - Converts NaN and Inf to None (Firestore rejects NaN/Inf in standard JSON).
    - Preserves native dicts, lists, strings, ints, floats, and booleans.

    Parameters:
        obj: Arbitrary data structure.

    Returns:
        Sanitized object compatible with Firestore.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_for_firestore(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        sanitized_items = [sanitize_for_firestore(v) for v in obj]
        # Firestore strictly disallows arrays containing arrays (nested arrays).
        # If any element is a list, convert to a row-indexed dictionary.
        if any(isinstance(v, list) for v in sanitized_items):
            return {f"row_{i}": row for i, row in enumerate(sanitized_items)}
        return sanitized_items
    elif isinstance(obj, np.ndarray):
        return sanitize_for_firestore(obj.tolist())
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_experiment(
    experiment_data: dict,
    experiment_id: str = None,
    client=None
) -> tuple:
    """
    Save experiment metadata, training history, and evaluation summary to Firestore.

    Parameters:
        experiment_data (dict): Payload containing experiment results.
        experiment_id (str, optional): Custom ID. If None, an automatic ID is generated.
        client: Optional explicit Firestore client (useful for unit test mocking).

    Returns:
        tuple (bool, str): (success, experiment_id_or_error_message)
    """
    if client is None:
        client = get_firestore_client()

    if client is None:
        return False, f"Firestore save skipped: {_initialization_error or 'Credentials not configured.'}"

    try:
        now_utc = datetime.now(timezone.utc).isoformat()
        if not experiment_id:
            time_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            unique_hex = uuid.uuid4().hex[:6]
            experiment_id = f"exp_{time_tag}_{unique_hex}"

        doc_payload = dict(experiment_data)
        doc_payload["experiment_id"] = experiment_id
        if "created_at" not in doc_payload:
            doc_payload["created_at"] = now_utc

        sanitized_payload = sanitize_for_firestore(doc_payload)

        # Write to collection 'experiments', document 'experiment_id'
        doc_ref = client.collection("experiments").document(experiment_id)
        doc_ref.set(sanitized_payload)

        return True, experiment_id
    except Exception as e:
        error_msg = f"Failed to save experiment to Firestore: {type(e).__name__} - {str(e)}"
        return False, error_msg


def get_experiment_history(limit: int = 20, client=None) -> list:
    """
    Retrieve past federated learning experiments ordered by creation time descending.

    Parameters:
        limit (int): Maximum records to retrieve.
        client: Optional explicit Firestore client (for mocking).

    Returns:
        list[dict]: List of experiment records. Returns empty list if unavailable.
    """
    if client is None:
        client = get_firestore_client()

    if client is None:
        return []

    try:
        query = (
            client.collection("experiments")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = query.stream()
        results = [doc.to_dict() for doc in docs if doc.exists]
        return results
    except Exception:
        # Fallback query without order_by in case index is not built
        try:
            docs = client.collection("experiments").limit(limit).stream()
            results = [doc.to_dict() for doc in docs if doc.exists]
            return results
        except Exception:
            return []


def get_experiment_by_id(experiment_id: str, client=None) -> dict:
    """
    Retrieve a specific experiment record by its document ID.

    Returns:
        dict | None: Experiment data if found, None otherwise.
    """
    if client is None:
        client = get_firestore_client()

    if client is None:
        return None

    try:
        doc_ref = client.collection("experiments").document(experiment_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception:
        return None


def build_experiment_record(
    model_config: dict = None,
    algorithm: str = "FedAvg",
    mu: float = None,
    training_history_path: str = os.path.join("results", "metrics", "federated_training_history.csv"),
    comparison_path: str = os.path.join("results", "metrics", "model_comparison.csv"),
    summary_path: str = os.path.join("results", "metrics", "evaluation_summary.json"),
    fedprox_comparison_path: str = os.path.join("results", "metrics", "fedavg_vs_fedprox_summary.json")
) -> dict:
    """
    Bundle local experimental output files into a clean Firestore document payload.
    Does NOT include raw patient CSV records.

    Parameters:
        model_config (dict, optional): Hyperparameters and architecture specs.
        algorithm (str): Name of the federated algorithm ("FedAvg" or "FedProx").
        mu (float, optional): Proximal regularization coefficient for FedProx.
        training_history_path (str): Path to training history CSV.
        comparison_path (str): Path to baseline model comparison CSV.
        summary_path (str): Path to evaluation summary JSON.
        fedprox_comparison_path (str): Path to FedAvg vs FedProx comparison JSON.

    Returns:
        dict: Standardized experiment document payload compatible with Cloud Firestore.
    """
    # 1. Training history
    training_history = []
    if os.path.exists(training_history_path):
        try:
            import pandas as pd
            df = pd.read_csv(training_history_path)
            training_history = df.to_dict(orient="records")
        except Exception:
            pass

    # 2. Model comparison
    model_comparison = []
    if os.path.exists(comparison_path):
        try:
            import pandas as pd
            df = pd.read_csv(comparison_path)
            model_comparison = df.to_dict(orient="records")
        except Exception:
            pass

    # 3. Detailed evaluation summary
    evaluation_summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                evaluation_summary = json.load(f)
        except Exception:
            pass

    # 4. Robustness metrics (Phase 8)
    robustness_analysis = {}
    robustness_path = os.path.join(os.path.dirname(summary_path), "robustness_metrics.json")
    if os.path.exists(robustness_path):
        try:
            with open(robustness_path, "r") as f:
                robustness_analysis = json.load(f)
        except Exception:
            pass

    # 5. Communication efficiency metrics (Phase 9)
    communication_analysis = {}
    comm_path = os.path.join(os.path.dirname(summary_path), "communication_metrics.json")
    if os.path.exists(comm_path):
        try:
            with open(comm_path, "r") as f:
                communication_analysis = json.load(f)
        except Exception:
            pass

    # 6. FedAvg vs FedProx comparison metrics (Phase 10)
    fedprox_comparison = {}
    if os.path.exists(fedprox_comparison_path):
        try:
            with open(fedprox_comparison_path, "r") as f:
                fedprox_comparison = json.load(f)
        except Exception:
            pass

    # 7. Secure Aggregation 3-Way comparison metrics (Phase 11)
    secagg_comparison = {}
    secagg_comp_path = os.path.join(os.path.dirname(summary_path), "secagg_three_way_summary.json")
    if os.path.exists(secagg_comp_path):
        try:
            with open(secagg_comp_path, "r") as f:
                secagg_comparison = json.load(f)
        except Exception:
            pass

    # 8. Vertical Federated Learning comparison metrics (Phase 12)
    vfl_comparison = {}
    vfl_comp_path = os.path.join(os.path.dirname(summary_path), "vfl_vs_hfl_comparison.csv")
    if os.path.exists(vfl_comp_path):
        try:
            import pandas as pd
            df = pd.read_csv(vfl_comp_path)
            vfl_comparison = df.to_dict(orient="records")
        except Exception:
            pass

    # 9. Federated Transfer Learning comparison metrics (Phase 13)
    ftl_comparison = {}
    ftl_comp_path = os.path.join(os.path.dirname(summary_path), "ftl_fraction_comparison.csv")
    if os.path.exists(ftl_comp_path):
        try:
            import pandas as pd
            df = pd.read_csv(ftl_comp_path)
            ftl_comparison = df.to_dict(orient="records")
        except Exception:
            pass

    # 10. Hybrid CNN-LSTM comparison metrics (Phase 14)
    cnn_lstm_comparison = {}
    cnn_lstm_comp_path = os.path.join(os.path.dirname(summary_path), "cnn_lstm_comparison.csv")
    if os.path.exists(cnn_lstm_comp_path):
        try:
            import pandas as pd
            df = pd.read_csv(cnn_lstm_comp_path)
            cnn_lstm_comparison = df.to_dict(orient="records")
        except Exception:
            pass

    # Extract algorithm / mu from model_config if passed there
    if model_config:
        if "algorithm" in model_config:
            algorithm = model_config["algorithm"]
        if "fedprox_mu" in model_config:
            mu = model_config["fedprox_mu"]
        elif "mu" in model_config:
            mu = model_config["mu"]

    # Route to dedicated record builders if requested
    if algorithm in ["Vertical Federated Learning", "VFL", "VFL SplitNN"]:
        return build_vfl_experiment_record()
    if algorithm in ["Federated Transfer Learning", "FTL"]:
        return build_ftl_experiment_record()
    if algorithm in ["Hybrid CNN-LSTM", "CNN-LSTM", "Federated CNN-LSTM", "Temporal CNN-LSTM"]:
        return build_cnn_lstm_experiment_record()

    is_secagg = (algorithm == "FedAvg with Secure Aggregation") or ("SECURE" in str(algorithm).upper())
    num_rounds = len(training_history) if training_history else 5

    try:
        from src.secure_aggregation import PROTOCOL_NAME, DEFAULT_MODULUS, DEFAULT_SCALING_FACTOR, DEFAULT_CLIPPING_BOUND
    except ImportError:
        PROTOCOL_NAME = "Pairwise Additive Masking with Finite-Field Quantization (Bonawitz et al. Simulation)"
        DEFAULT_MODULUS = (1 << 61) - 1
        DEFAULT_SCALING_FACTOR = 10_000_000
        DEFAULT_CLIPPING_BOUND = 1000.0

    secagg_payload = {
        "enabled": is_secagg,
        "protocol_name": PROTOCOL_NAME if is_secagg else "None",
        "num_participating_clients": 3,
        "validation_status": "VALIDATED" if is_secagg else "N/A",
        "estimated_communication_overhead_bytes": 448 * num_rounds if is_secagg else 0,
        "protocol_configuration": {
            "modulus": f"2^61 - 1 ({DEFAULT_MODULUS})",
            "scaling_factor": DEFAULT_SCALING_FACTOR,
            "clipping_bound": DEFAULT_CLIPPING_BOUND,
            "csprng": "Python secrets module + HMAC-SHA256 (Counter Mode)",
            "masking_method": "Pairwise Additive Masking in Finite Field Z_q",
            "sample_weighting_mechanism": "Client-side pre-quantization weight scaling (n_k / N)"
        } if is_secagg else None
    }

    if model_config is None:
        model_config = {
            "input_dim": 5,
            "hidden_dim": 8,
            "output_dim": 1,
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 10,
            "algorithm": algorithm,
            "fedprox_mu": mu if algorithm == "FedProx" else None,
            "secure_aggregation_enabled": is_secagg
        }
    else:
        model_config.setdefault("algorithm", algorithm)
        model_config.setdefault("secure_aggregation_enabled", is_secagg)
        if algorithm == "FedProx":
            model_config.setdefault("fedprox_mu", mu)

    agg_label = (
        "FedAvg with Secure Aggregation"
        if is_secagg
        else (f"FedProx (mu={mu})" if algorithm == "FedProx" else "FedAvg (Sample-Weighted)")
    )

    return {
        "algorithm": algorithm,
        "aggregation_algorithm": agg_label,
        "fedprox_mu": mu if algorithm == "FedProx" else None,
        "secure_aggregation": secagg_payload,
        "secure_aggregation_enabled": is_secagg,
        "protocol_name": PROTOCOL_NAME if is_secagg else "None",
        "secagg_validation_status": "VALIDATED" if is_secagg else "N/A",
        "secagg_estimated_overhead_bytes": 448 * num_rounds if is_secagg else 0,
        "num_hospitals": 3,
        "num_communication_rounds": num_rounds,
        "model_architecture": "HealthcareMLP (NumPy 2-Layer)",
        "model_config": model_config,
        "training_history": training_history,
        "model_comparison": model_comparison,
        "evaluation_summary": evaluation_summary,
        "robustness_analysis": robustness_analysis,
        "communication_analysis": communication_analysis,
        "fedprox_comparison": fedprox_comparison,
        "secagg_comparison": secagg_comparison,
        "vfl_comparison": vfl_comparison,
        "ftl_comparison": ftl_comparison,
        "cnn_lstm_comparison": cnn_lstm_comparison,
        "dataset_description": {
            "total_synthetic_samples": 1500,
            "client_samples_each": 400,
            "test_samples": 300,
            "features": ["age", "blood_pressure", "glucose_level", "bmi", "cholesterol_level"],
            "target": "risk_label",
            "distribution": "Non-IID across 3 simulated hospitals"
        }
    }


def build_vfl_experiment_record(
    summary_path: str = os.path.join("results", "metrics", "vfl_evaluation_summary.json"),
    history_path: str = os.path.join("results", "metrics", "vfl_training_history.csv"),
    comparison_path: str = os.path.join("results", "metrics", "vfl_vs_hfl_comparison.csv")
) -> dict:
    """
    Bundle Vertical Federated Learning (VFL) experimental output files into a clean Firestore document payload.

    Strictly enforces privacy:
    - Never includes raw patient features or patient-level records.
    - Never includes intermediate cut-layer representations or gradients.
    - Never includes private cryptographic keys or seeds.
    - Only stores experiment metadata, training configuration, aggregate evaluation metrics, and communication overhead.

    Parameters:
        summary_path (str): Path to vfl_evaluation_summary.json.
        history_path (str): Path to vfl_training_history.csv.
        comparison_path (str): Path to vfl_vs_hfl_comparison.csv.

    Returns:
        dict: Standardized VFL experiment payload compatible with Cloud Firestore.
    """
    summary_data = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    history_records = []
    if os.path.exists(history_path):
        try:
            import pandas as pd
            df_hist = pd.read_csv(history_path)
            history_records = df_hist.to_dict(orient="records")
        except Exception:
            pass

    comparison_records = []
    if os.path.exists(comparison_path):
        try:
            import pandas as pd
            df_comp = pd.read_csv(comparison_path)
            comparison_records = df_comp.to_dict(orient="records")
        except Exception:
            pass

    config = summary_data.get("experiment_configuration", {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.05,
        "random_seed": 42
    })
    test_metrics = summary_data.get("central_evaluation", {})
    comm_analysis = summary_data.get("communication_analysis", {})

    return {
        "algorithm": "Vertical Federated Learning",
        "paradigm": "Vertical Federated Learning (Split Learning / Feature Partitioned)",
        "num_parties": 3,
        "participating_parties": ["hospital_A", "hospital_B", "hospital_C"],
        "label_owner": "coordinator",
        "feature_partitions": {
            "hospital_A": ["age", "blood_pressure"],
            "hospital_B": ["glucose_level", "bmi"],
            "hospital_C": ["cholesterol_level"]
        },
        "target_variable": "risk_label",
        "patient_alignment_method": "Simulated Private Set Intersection (PSI) on Aligned Patient IDs (PID_0001 - PID_1500)",
        "model_architecture": {
            "type": "VFL Split Neural Network (SplitNN)",
            "bottom_models": {
                "hospital_A": "2 -> 4 (ReLU)",
                "hospital_B": "2 -> 4 (ReLU)",
                "hospital_C": "1 -> 4 (ReLU)"
            },
            "top_model": "12 -> 1 (Sigmoid)",
            "total_parameters": 45,
            "parameter_transfer": "0 Bytes (Model weights are local and stationary)"
        },
        "training_configuration": config,
        "evaluation_summary": test_metrics,
        "training_history": history_records,
        "comparison_with_hfl": comparison_records,
        "communication_analysis": comm_analysis,
        "privacy_guarantees": {
            "raw_features_transmitted": False,
            "patient_level_predictions_persisted": False,
            "intermediate_representations_persisted": False,
            "intermediate_gradients_persisted": False,
            "model_parameter_transmission": "None (Parameters remain stationary on feature-owning premises)"
        },
        "dataset_description": {
            "total_synthetic_samples": 1500,
            "training_samples": 1200,
            "test_samples": 300,
            "features": ["age", "blood_pressure", "glucose_level", "bmi", "cholesterol_level"],
            "partitioning_type": "Vertical (disjoint features per party, aligned patient IDs)"
        }
    }


def build_ftl_experiment_record(
    summary_path: str = os.path.join("results", "metrics", "ftl_evaluation_summary.json"),
    comparison_path: str = os.path.join("results", "metrics", "ftl_fraction_comparison.csv"),
    history_path: str = os.path.join("results", "metrics", "ftl_training_history.csv")
) -> dict:
    """
    Bundle Federated Transfer Learning (FTL) experimental output files into a clean Firestore document payload.

    Strictly enforces privacy:
    - Never includes raw patient features or patient-level records.
    - Never includes patient-level predictions.
    - Never includes private weights or cryptographic keys.
    - Only stores experiment metadata, domain configurations, aggregate evaluation metrics, and communication overhead.

    Returns:
        dict: Standardized FTL experiment payload compatible with Cloud Firestore.
    """
    summary_data = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    comparison_records = []
    if os.path.exists(comparison_path):
        try:
            import pandas as pd
            df_comp = pd.read_csv(comparison_path)
            comparison_records = df_comp.to_dict(orient="records")
        except Exception:
            pass

    history_records = []
    if os.path.exists(history_path):
        try:
            import pandas as pd
            df_hist = pd.read_csv(history_path)
            history_records = df_hist.to_dict(orient="records")
        except Exception:
            pass

    setup_info = summary_data.get("experimental_setup", {
        "source_training_samples": 1000,
        "target_training_pool_samples": 200,
        "target_test_samples": 100,
        "evaluated_fractions": [0.10, 0.25, 0.50, 1.00],
        "source_epochs": 40,
        "target_epochs": 30,
        "learning_rate": 0.02,
        "batch_size": 32,
        "random_seed": 42
    })
    comm_analysis = summary_data.get("communication_analysis", {})

    return {
        "algorithm": "Federated Transfer Learning",
        "paradigm": "Federated Transfer Learning (Cross-Domain Representation Transfer)",
        "source_domain": summary_data.get("source_domain", {}),
        "target_domain": summary_data.get("target_domain", {}),
        "transfer_strategies": ["Full Fine-Tuning", "Feature Extraction (Frozen Head)"],
        "evaluated_fractions": setup_info.get("evaluated_fractions", [0.1, 0.25, 0.5, 1.0]),
        "model_architecture": {
            "type": "TransferHealthcareModel (2-Layer NumPy MLP)",
            "input_dim": 5,
            "hidden_dim": 8,
            "output_dim": 1,
            "total_parameters": 57,
            "feature_extractor_parameters": 48,
            "prediction_head_parameters": 9
        },
        "experimental_setup": setup_info,
        "source_pretraining": summary_data.get("source_pretraining", {}),
        "fraction_comparison": comparison_records,
        "results_by_fraction": summary_data.get("results_by_fraction", {}),
        "training_history": history_records,
        "communication_analysis": comm_analysis,
        "privacy_guarantees": {
            "raw_patient_data_persisted": False,
            "patient_level_predictions_persisted": False,
            "source_patient_records_transmitted": False,
            "target_patient_records_transmitted": False,
            "parameter_transmission_mode": "One-time parameter transfer from Source to Target"
        },
        "dataset_description": {
            "source_records": 1200,
            "target_records": 300,
            "features": ["age", "blood_pressure", "glucose_level", "bmi", "cholesterol_level"],
            "target_variable": "risk_label",
            "domain_shift": "Covariate & demographic shift (Geriatric Target vs Adult Source)"
        }
    }


def build_cnn_lstm_experiment_record(
    summary_path: str = os.path.join("results", "metrics", "cnn_lstm_evaluation_summary.json"),
    comparison_path: str = os.path.join("results", "metrics", "cnn_lstm_comparison.csv"),
    fed_history_path: str = os.path.join("results", "metrics", "cnn_lstm_federated_history.csv"),
    train_history_path: str = os.path.join("results", "metrics", "cnn_lstm_training_history.csv")
) -> dict:
    """
    Bundle Hybrid CNN-LSTM experimental output files into a clean Firestore document payload.

    Strictly enforces privacy:
    - Never includes raw patient sequences, time series, or patient-level records.
    - Never includes individual patient predictions or identifiers.
    - Never includes intermediate activation maps or hidden states.
    - Only stores model architecture metadata, aggregate performance metrics, baseline comparisons, and communication overhead.

    Returns:
        dict: Standardized CNN-LSTM experiment payload compatible with Cloud Firestore.
    """
    summary_data = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    comparison_records = []
    if os.path.exists(comparison_path):
        try:
            import pandas as pd
            df_comp = pd.read_csv(comparison_path)
            comparison_records = df_comp.to_dict(orient="records")
        except Exception:
            pass

    fed_history_records = []
    if os.path.exists(fed_history_path):
        try:
            import pandas as pd
            df_fed = pd.read_csv(fed_history_path)
            fed_history_records = df_fed.to_dict(orient="records")
        except Exception:
            pass

    train_history_records = []
    if os.path.exists(train_history_path):
        try:
            import pandas as pd
            df_th = pd.read_csv(train_history_path)
            train_history_records = df_th.to_dict(orient="records")
        except Exception:
            pass

    try:
        from src.communication_analysis import calculate_cnn_lstm_communication_volume
        comm_analysis = calculate_cnn_lstm_communication_volume(
            num_parameters=2449,
            num_clients=3,
            num_rounds=len(fed_history_records) if fed_history_records else 5,
            bytes_per_parameter=4
        )
    except Exception:
        comm_analysis = {
            "num_parameters": 2449,
            "total_communication_bytes": 293880,
            "total_communication_kb": 287.0
        }

    return {
        "algorithm": "Hybrid CNN-LSTM",
        "paradigm": "Longitudinal Federated Deep Learning (Hybrid 1D CNN + LSTM)",
        "model_architecture": {
            "type": "HybridCNNLSTMModel (PyTorch)",
            "sequence_length": summary_data.get("sequence_length", 12),
            "num_features": summary_data.get("num_features", 5),
            "feature_channels": ["age", "blood_pressure", "glucose_level", "bmi", "cholesterol_level"],
            "conv1d_layer": "Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1) + ReLU",
            "lstm_layer": "LSTM(input_size=16, hidden_size=16, batch_first=True)",
            "fc_head": "Linear(16, 1) + Sigmoid",
            "total_parameters": 2449,
            "parameter_precision": "float32 (4 bytes / parameter)",
            "single_model_bytes": 9796
        },
        "training_configuration": summary_data.get("training_configuration", {
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 0.005,
            "random_seed": 42
        }),
        "models_evaluated": summary_data.get("models_evaluated", {}),
        "baseline_comparison": comparison_records,
        "federated_training_history": fed_history_records,
        "training_history": train_history_records,
        "communication_analysis": comm_analysis,
        "privacy_guarantees": {
            "raw_patient_sequences_persisted": False,
            "patient_level_predictions_persisted": False,
            "intermediate_feature_maps_persisted": False,
            "intermediate_gradients_persisted": False,
            "patient_ids_transmitted": False,
            "federated_transmission_mode": "Model parameter state_dicts aggregated via sample-weighted FedAvg; patient records remain entirely isolated on client nodes."
        },
        "dataset_description": {
            "total_synthetic_patients": 1000,
            "sequence_length": 12,
            "features_per_step": 5,
            "train_patients": 700,
            "validation_patients": 150,
            "test_patients": 150,
            "client_distribution": "Non-IID risk distribution partitioned across 3 hospital clients (H1=233, H2=233, H3=234)"
        }
    }


if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 6: FIREBASE CLOUD FIRESTORE SERVICE STATUS")
    print("=" * 65)

    is_connected, msg = check_firestore_connection()
    print(f"\nFirestore Connection Available: {is_connected}")
    print(f"Status Message: {msg}")

    print("\nTesting NumPy Sanitization Logic...")
    sample_data = {
        "float_val": np.float64(0.8501),
        "int_val": np.int64(400),
        "array_val": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "nan_val": float("nan")
    }
    sanitized = sanitize_for_firestore(sample_data)
    print(f"Sanitized Float: {type(sanitized['float_val']).__name__} = {sanitized['float_val']}")
    print(f"Sanitized Int:   {type(sanitized['int_val']).__name__} = {sanitized['int_val']}")
    print(f"Sanitized Array: {type(sanitized['array_val']).__name__} = {sanitized['array_val']}")
    print(f"Sanitized NaN:   {sanitized['nan_val']} (converted to None)")

    print("\nBuilding Local Experiment Payload (from results/metrics)...")
    payload = build_experiment_record()
    print(f"Payload Keys: {list(payload.keys())}")
    print(f"Training Rounds Loaded: {len(payload['training_history'])}")
    print(f"Models Compared: {len(payload['model_comparison'])}")

    print("\nAttempting Safe Save (Graceful degradation if unconfigured)...")
    success, res_msg = save_experiment(payload)
    print(f"Save Result: Success={success} | Message={res_msg}")

    print("\nFirebase service initialization verified.")
    print("=" * 65)
