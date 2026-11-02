"""
Unit tests for Hospital Client Simulation (src/client.py).
"""

import unittest
import os
import numpy as np

from src.client import HospitalClient
from src.model import HealthcareMLP
from src.data_preparation import load_scaler


class TestHospitalClient(unittest.TestCase):
    """Test suite for HospitalClient isolated training and privacy boundary."""

    @classmethod
    def setUpClass(cls):
        cls.scaler_path = os.path.join("data", "scaler_params.json")
        cls.scaler = load_scaler(cls.scaler_path)

        # Baseline shared initial model weights
        cls.base_model = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=42)
        cls.initial_params = cls.base_model.get_parameters()

    def test_client_initialization_and_samples(self):
        """Verify client loads its designated local file (400 samples)."""
        client = HospitalClient(
            hospital_id="hospital_1",
            data_path=os.path.join("data", "hospital_1_train.csv"),
            scaler=self.scaler
        )
        self.assertEqual(client.hospital_id, "hospital_1")
        self.assertEqual(client.num_samples, 400)
        self.assertEqual(client.X_train.shape, (400, 5))
        self.assertEqual(client.y_train.shape, (400, 1))

    def test_all_three_clients_independent_training(self):
        """Verify all 3 hospitals train independently and return identical parameter structure."""
        hospital_ids = ["hospital_1", "hospital_2", "hospital_3"]
        trained_summaries = []

        for h_id in hospital_ids:
            client = HospitalClient(
                hospital_id=h_id,
                data_path=os.path.join("data", f"{h_id}_train.csv"),
                scaler=self.scaler
            )
            # Load initial global parameters
            client.set_model_parameters(self.initial_params)

            # Perform local training
            summary = client.train(epochs=5, batch_size=32, learning_rate=0.01)
            trained_summaries.append(summary)

            # Assert returned keys
            self.assertIn("hospital_id", summary)
            self.assertIn("num_samples", summary)
            self.assertIn("initial_loss", summary)
            self.assertIn("final_loss", summary)
            self.assertIn("loss_history", summary)
            self.assertIn("parameters", summary)

            # Check loss is finite and valid
            self.assertFalse(np.isnan(summary["initial_loss"]))
            self.assertFalse(np.isnan(summary["final_loss"]))

            # Verify parameter shapes
            params = summary["parameters"]
            self.assertEqual(set(params.keys()), {"W1", "b1", "W2", "b2"})
            self.assertEqual(params["W1"].shape, (5, 8))
            self.assertEqual(params["b1"].shape, (1, 8))
            self.assertEqual(params["W2"].shape, (8, 1))
            self.assertEqual(params["b2"].shape, (1, 1))

            # Parameters must have changed from initial parameters
            w1_delta = np.sum(np.abs(params["W1"] - self.initial_params["W1"]))
            self.assertGreater(w1_delta, 1e-5)

        # Confirm client parameter updates diverge due to Non-IID data
        h1_w1 = trained_summaries[0]["parameters"]["W1"]
        h2_w1 = trained_summaries[1]["parameters"]["W1"]
        self.assertFalse(np.allclose(h1_w1, h2_w1), "Hospitals with Non-IID data should produce distinct local weights")

    def test_no_raw_data_leakage_in_return_payload(self):
        """Client train() must only return model parameters and aggregates, never raw patient records."""
        client = HospitalClient(
            hospital_id="hospital_1",
            data_path=os.path.join("data", "hospital_1_train.csv"),
            scaler=self.scaler
        )
        summary = client.train(epochs=2, batch_size=32, learning_rate=0.01)

        # Ensure no dataframe or feature matrix is in the return payload
        self.assertNotIn("raw_data", summary)
        self.assertNotIn("X_train", summary)
        self.assertNotIn("y_train", summary)
        self.assertNotIn("records", summary)

    def test_local_evaluation(self):
        """Verify local evaluation returns valid loss and accuracy."""
        client = HospitalClient(
            hospital_id="hospital_2",
            data_path=os.path.join("data", "hospital_2_train.csv"),
            scaler=self.scaler
        )
        eval_metrics = client.evaluate_local()
        self.assertEqual(eval_metrics["hospital_id"], "hospital_2")
        self.assertEqual(eval_metrics["num_samples"], 400)
        self.assertGreaterEqual(eval_metrics["accuracy"], 0.0)
        self.assertLessEqual(eval_metrics["accuracy"], 1.0)
        self.assertFalse(np.isnan(eval_metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
