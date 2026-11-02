"""
Unit tests for Federated Experiment Loop and Central Coordinator (src/experiment.py).
"""

import unittest
import os
import pandas as pd
import numpy as np

from src.experiment import run_federated_experiment, initialize_hospital_clients
from src.data_preparation import load_scaler
from src.model import HealthcareMLP


class TestFederatedExperiment(unittest.TestCase):
    """Test suite for multi-round Federated Learning experiment coordination."""

    @classmethod
    def setUpClass(cls):
        cls.scaler_path = os.path.join("data", "scaler_params.json")
        cls.scaler = load_scaler(cls.scaler_path)

    def test_client_initialization_all_three_hospitals(self):
        """Verify all three hospitals are properly initialized with local data."""
        clients = initialize_hospital_clients(scaler=self.scaler, random_seed=42)
        self.assertEqual(set(clients.keys()), {"hospital_1", "hospital_2", "hospital_3"})

        for h_id, client in clients.items():
            self.assertEqual(client.hospital_id, h_id)
            self.assertEqual(client.num_samples, 400)
            self.assertEqual(client.X_train.shape, (400, 5))
            self.assertEqual(client.y_train.shape, (400, 1))

    def test_run_federated_experiment_execution(self):
        """Run a 3-round experiment to verify end-to-end multi-round execution."""
        test_history_path = os.path.join("results", "metrics", "test_fed_history.csv")

        global_model, history_df = run_federated_experiment(
            num_rounds=3,
            local_epochs=3,
            batch_size=32,
            learning_rate=0.01,
            random_seed=42,
            save_history=True,
            history_output_path=test_history_path
        )

        # 1. Verify global model is valid instance with correct parameter shapes
        self.assertIsInstance(global_model, HealthcareMLP)
        params = global_model.get_parameters()
        self.assertEqual(params["W1"].shape, (5, 8))
        self.assertEqual(params["b1"].shape, (1, 8))
        self.assertEqual(params["W2"].shape, (8, 1))
        self.assertEqual(params["b2"].shape, (1, 1))

        # 2. Verify number of executed rounds
        self.assertEqual(len(history_df), 3)
        self.assertEqual(list(history_df["round"]), [1, 2, 3])

        # 3. Verify history columns
        expected_cols = {
            "round",
            "hospital_1_loss",
            "hospital_2_loss",
            "hospital_3_loss",
            "average_local_loss"
        }
        self.assertTrue(expected_cols.issubset(set(history_df.columns)))

        # 4. Verify losses are finite numbers
        for col in ["hospital_1_loss", "hospital_2_loss", "hospital_3_loss", "average_local_loss"]:
            self.assertFalse(history_df[col].isnull().any())
            self.assertFalse(np.isinf(history_df[col].to_numpy()).any())

        # 5. Check CSV file was written
        self.assertTrue(os.path.exists(test_history_path))
        loaded_df = pd.read_csv(test_history_path)
        self.assertEqual(len(loaded_df), 3)

        # Clean up temporary test file
        if os.path.exists(test_history_path):
            os.remove(test_history_path)

    def test_clients_receive_identical_parameters_at_start_of_round(self):
        """Assert that coordinator broadcasts the identical global weights to all clients."""
        clients = initialize_hospital_clients(scaler=self.scaler, random_seed=42)
        base_model = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=100)
        global_weights = base_model.get_parameters()

        # Broadcast to all clients
        for client in clients.values():
            client.set_model_parameters(global_weights)

        # Verify initial weights match across all clients
        h1_weights = clients["hospital_1"].get_model_parameters()
        h2_weights = clients["hospital_2"].get_model_parameters()
        h3_weights = clients["hospital_3"].get_model_parameters()

        for key in ["W1", "b1", "W2", "b2"]:
            self.assertTrue(np.array_equal(h1_weights[key], global_weights[key]))
            self.assertTrue(np.array_equal(h2_weights[key], global_weights[key]))
            self.assertTrue(np.array_equal(h3_weights[key], global_weights[key]))


if __name__ == "__main__":
    unittest.main()
