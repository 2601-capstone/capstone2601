"""
Hospital Client Node Simulation for Federated Healthcare Analytics.

This module simulates an independent hospital node in a Federated Learning network:
1. Loads strictly its own private local clinical dataset.
2. Applies shared feature standardization to operate in a unified feature space.
3. Receives global model parameters from a coordinator.
4. Performs local model training using mini-batch gradient descent.
5. Returns updated model parameters without exposing private patient records.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root directory is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import HealthcareMLP
from src.data_preparation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataset,
    apply_scaler,
    load_scaler
)


class HospitalClient:
    """
    Simulates an isolated hospital node participating in Federated Learning.
    Each client owns private clinical records and performs local model optimization.
    """

    def __init__(
        self,
        hospital_id: str,
        data_path: str,
        scaler: dict = None,
        model_config: dict = None
    ):
        """
        Initialize the Hospital Client.

        Parameters:
            hospital_id (str): Unique identifier (e.g. 'hospital_1').
            data_path (str): Path to local training CSV file.
            scaler (dict): Pre-computed feature standardization parameters.
            model_config (dict): Architecture options (input_dim, hidden_dim, seed).
        """
        self.hospital_id = str(hospital_id)
        self.data_path = data_path

        # Load shared scaler if not directly supplied
        if scaler is None:
            scaler_file = os.path.join("data", "scaler_params.json")
            if os.path.exists(scaler_file):
                self.scaler = load_scaler(scaler_file)
            else:
                self.scaler = None
        else:
            self.scaler = scaler

        # Default model configuration
        self.model_config = {
            "input_dim": 5,
            "hidden_dim": 8,
            "output_dim": 1,
            "random_seed": 42
        }
        if model_config is not None:
            self.model_config.update(model_config)

        # Initialize local neural network
        self.model = HealthcareMLP(
            input_dim=self.model_config["input_dim"],
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
            random_seed=self.model_config["random_seed"]
        )

        # Data placeholders (private to client)
        self.X_train = None
        self.y_train = None
        self.num_samples = 0
        self.global_parameters = None

        # Load and preprocess local data
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self) -> None:
        """
        Load private local dataset from CSV and apply feature standardization.
        Guarantees that raw patient records stay inside the client.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[{self.hospital_id}] Local dataset not found at: {self.data_path}")

        raw_df = pd.read_csv(self.data_path)
        validate_dataset(raw_df)

        self.num_samples = len(raw_df)

        # Apply shared standardization to feature columns
        if self.scaler is not None:
            self.X_train = apply_scaler(raw_df, self.scaler)
        else:
            # Fallback if no scaler provided
            self.X_train = raw_df[FEATURE_COLUMNS].to_numpy(dtype=float)

        self.y_train = raw_df[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)

    def set_model_parameters(self, parameters: dict) -> None:
        """
        Load parameters (e.g., received from central coordinator) into the local model
        and retain a local copy of global parameters for FedProx proximal regularization.
        """
        self.model.set_parameters(parameters)
        self.global_parameters = {k: np.asarray(v).copy() for k, v in parameters.items()}

    def get_model_parameters(self) -> dict:
        """
        Return a copy of the local model parameters (weights and biases).
        No raw data rows are exposed.
        """
        return self.model.get_parameters()

    def train(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = False,
        mu: float = 0.0,
        global_parameters: dict = None
    ) -> dict:
        """
        Train the local model on private hospital data using mini-batch gradient descent,
        with optional FedProx proximal regularization:
            L(w) = Loss_BCE(w) + (mu / 2) * ||w - w_t||^2

        Parameters:
            epochs (int): Number of local training epochs.
            batch_size (int): Size of mini-batches.
            learning_rate (float): Step size for gradient descent.
            verbose (bool): Whether to log progress.
            mu (float): FedProx proximal coefficient (mu >= 0).
            global_parameters (dict): Reference global parameters w_t. If None, uses retained copy.

        Returns:
            dict: Training summary containing:
                - hospital_id (str)
                - num_samples (int)
                - initial_loss (float)
                - final_loss (float)
                - loss_history (list[float])
                - parameters (dict of np.ndarray)
                - mu (float)
        """
        if mu < 0.0:
            raise ValueError(f"Proximal coefficient mu must be non-negative, got {mu}.")

        if self.X_train is None or self.y_train is None:
            raise RuntimeError(f"[{self.hospital_id}] Local data is not loaded.")

        # Resolve global reference parameters for FedProx
        ref_global = global_parameters if global_parameters is not None else self.global_parameters

        # Compute initial loss before training
        initial_probs = self.model.predict_proba(self.X_train)
        initial_loss = self.model.compute_loss(self.y_train, initial_probs)
        if mu > 0.0 and ref_global is not None:
            initial_loss += self.model.compute_proximal_loss(self.model.parameters, ref_global, mu=mu)

        # Fit model on local data
        loss_history = self.model.fit(
            X=self.X_train,
            y=self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
            global_parameters=ref_global,
            mu=mu
        )

        final_loss = loss_history[-1] if loss_history else initial_loss

        return {
            "hospital_id": self.hospital_id,
            "num_samples": self.num_samples,
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "loss_history": loss_history,
            "parameters": self.get_model_parameters(),
            "mu": float(mu)
        }

    def evaluate_local(self) -> dict:
        """
        Evaluate local model performance on private hospital data.

        Returns:
            dict: Local loss and accuracy metrics.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError(f"[{self.hospital_id}] Local data is not loaded.")

        probs = self.model.predict_proba(self.X_train)
        loss = self.model.compute_loss(self.y_train, probs)
        preds = (probs >= 0.5).astype(int)
        accuracy = float(np.mean(preds == self.y_train))

        return {
            "hospital_id": self.hospital_id,
            "num_samples": self.num_samples,
            "loss": float(loss),
            "accuracy": float(accuracy)
        }


if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 3: HOSPITAL CLIENT SIMULATION & LOCAL TRAINING")
    print("=" * 65)

    # 1. Load shared scaler
    scaler_path = os.path.join("data", "scaler_params.json")
    shared_scaler = load_scaler(scaler_path)
    print(f"Loaded shared scaler from: {scaler_path}")

    # 2. Initialize initial global model parameters
    initial_model = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=42)
    initial_weights = initial_model.get_parameters()
    print("Created shared initial model weights.")

    # 3. Simulate training across 3 hospitals
    hospital_ids = ["hospital_1", "hospital_2", "hospital_3"]
    client_results = []

    for h_id in hospital_ids:
        csv_path = os.path.join("data", f"{h_id}_train.csv")
        client = HospitalClient(
            hospital_id=h_id,
            data_path=csv_path,
            scaler=shared_scaler
        )

        # Set shared initial weights
        client.set_model_parameters(initial_weights)

        # Perform local training
        result = client.train(epochs=10, batch_size=32, learning_rate=0.01)
        client_results.append(result)

        print(f"\n[{h_id.upper()}] Local Training Summary:")
        print(f"   Private Samples: {result['num_samples']}")
        print(f"   Initial Loss:    {result['initial_loss']:.4f}")
        print(f"   Final Loss:      {result['final_loss']:.4f}")
        print(f"   Loss Delta:      {result['initial_loss'] - result['final_loss']:.4f}")
        print(f"   Exported Param Shapes: {[f'{k}: {v.shape}' for k, v in result['parameters'].items()]}")

    print("\nLocal hospital training simulation completed successfully.")
    print("=" * 65)
