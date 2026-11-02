"""
Local Machine Learning Model Architecture for Healthcare Analytics Simulation.

This module implements a lightweight multi-layer perceptron (MLP) using pure NumPy:
- Input Features: 5 (Age, Blood Pressure, Glucose, BMI, Cholesterol)
- Hidden Layer: 8 Neurons with ReLU activation
- Output Layer: 1 Neuron with Sigmoid activation
- Loss Function: Binary Cross-Entropy (BCE)
- Optimizer: Mini-batch Gradient Descent

Designed for clean local hospital client training and future Federated Averaging (FedAvg).
"""

import numpy as np


class HealthcareMLP:
    """
    Lightweight 2-layer Neural Network for binary healthcare risk classification.
    Implemented in NumPy for clarity and easy parameter aggregation in Federated Learning.
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

        # Initialize trainable parameters
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Initialize network weights using He (Kaiming) normal initialization
        and biases to zero for reproducible starting conditions.
        """
        rng = np.random.RandomState(self.random_seed)

        # Layer 1: (input_dim -> hidden_dim)
        w1_scale = np.sqrt(2.0 / self.input_dim)
        self.parameters["W1"] = rng.randn(self.input_dim, self.hidden_dim) * w1_scale
        self.parameters["b1"] = np.zeros((1, self.hidden_dim))

        # Layer 2: (hidden_dim -> output_dim)
        w2_scale = np.sqrt(2.0 / self.hidden_dim)
        self.parameters["W2"] = rng.randn(self.hidden_dim, self.output_dim) * w2_scale
        self.parameters["b2"] = np.zeros((1, self.output_dim))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit activation function."""
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function."""
        return (z > 0.0).astype(float)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid activation function.
        Clips extreme values to prevent overflow in exp(-z).
        """
        z_clipped = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def forward(self, X: np.ndarray) -> tuple:
        """
        Perform forward propagation.

        Parameters:
            X (np.ndarray): Input feature matrix of shape (N, input_dim).

        Returns:
            tuple: (A2, cache)
                - A2: Predicted probabilities of shape (N, 1).
                - cache: Intermediate tensors for backpropagation (X, Z1, A1, Z2, A2).
        """
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        # Hidden Layer
        Z1 = np.dot(X, W1) + b1
        A1 = self.relu(Z1)

        # Output Layer
        Z2 = np.dot(A1, W2) + b2
        A2 = self.sigmoid(Z2)

        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate binary cross-entropy loss.

        Parameters:
            y_true (np.ndarray): Ground truth labels (0 or 1), shape (N, 1) or (N,).
            y_pred (np.ndarray): Predicted probabilities, shape (N, 1).

        Returns:
            float: Average binary cross-entropy loss.
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)

        # Numerical stability clipping to avoid log(0)
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        bce = -(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
        return float(np.mean(bce))

    def backward(self, cache: tuple, y_true: np.ndarray) -> dict:
        """
        Perform backpropagation to compute analytical gradients for all parameters.

        Parameters:
            cache (tuple): Forward propagation cache (X, Z1, A1, Z2, A2).
            y_true (np.ndarray): Ground truth binary labels, shape (N, 1).

        Returns:
            dict: Gradients {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}.
        """
        X, Z1, A1, _, A2 = cache
        N = X.shape[0]
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)

        # Output layer gradient for Sigmoid + Binary Cross-Entropy
        dZ2 = (A2 - y_true) / N
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer backpropagation
        dA1 = np.dot(dZ2, self.parameters["W2"].T)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2
        }

    @staticmethod
    def compute_proximal_loss(parameters: dict, global_parameters: dict, mu: float = 0.0) -> float:
        """
        Calculate FedProx proximal regularization penalty:
            (mu / 2) * sum_{k} ||w_k - w_{t, k}||^2

        Parameters:
            parameters (dict): Current local model parameters.
            global_parameters (dict): Initial global model parameters at start of round.
            mu (float): Proximal regularization coefficient (mu >= 0).

        Returns:
            float: Proximal penalty value.
        """
        if mu < 0.0:
            raise ValueError(f"Proximal coefficient mu must be non-negative, got {mu}.")
        if mu == 0.0 or global_parameters is None:
            return 0.0

        prox_sum = 0.0
        for key in parameters:
            if key in global_parameters:
                diff = np.asarray(parameters[key], dtype=float) - np.asarray(global_parameters[key], dtype=float)
                prox_sum += float(np.sum(diff ** 2))

        return float(0.5 * mu * prox_sum)

    def update_parameters(self, gradients: dict, learning_rate: float = 0.01) -> None:
        """Update model parameters using gradient descent."""
        for key in self.parameters:
            self.parameters[key] -= learning_rate * gradients[key]

    def train_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        learning_rate: float = 0.01,
        global_parameters: dict = None,
        mu: float = 0.0
    ) -> float:
        """
        Perform one forward-backward training step on a batch and return loss.
        Supports optional FedProx proximal regularization:
            Loss = BCE(y, y_hat) + (mu / 2) * ||w - w_t||^2
            Gradient = Grad_BCE + mu * (w - w_t)
        """
        if mu < 0.0:
            raise ValueError(f"Proximal coefficient mu must be non-negative, got {mu}.")

        y_pred, cache = self.forward(X_batch)
        task_loss = self.compute_loss(y_batch, y_pred)
        gradients = self.backward(cache, y_batch)

        # Apply FedProx proximal gradient if mu > 0 and global_parameters is provided
        total_loss = task_loss
        if mu > 0.0 and global_parameters is not None:
            for key in self.parameters:
                if key in global_parameters:
                    diff = np.asarray(self.parameters[key], dtype=float) - np.asarray(global_parameters[key], dtype=float)
                    gradients[key] = gradients[key] + (mu * diff)
            total_loss += self.compute_proximal_loss(self.parameters, global_parameters, mu=mu)

        self.update_parameters(gradients, learning_rate=learning_rate)
        return total_loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = False,
        global_parameters: dict = None,
        mu: float = 0.0
    ) -> list:
        """
        Train the model using mini-batch gradient descent with optional FedProx proximal term.

        Returns:
            list[float]: Epoch-wise average training losses.
        """
        if mu < 0.0:
            raise ValueError(f"Proximal coefficient mu must be non-negative, got {mu}.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        num_samples = X.shape[0]

        loss_history = []
        rng = np.random.RandomState(self.random_seed)

        for epoch in range(epochs):
            # Shuffle indices at the start of each epoch
            indices = rng.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            batch_losses = []
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss = self.train_step(
                    X_batch,
                    y_batch,
                    learning_rate=learning_rate,
                    global_parameters=global_parameters,
                    mu=mu
                )
                batch_losses.append(batch_loss)

            epoch_loss = float(np.mean(batch_losses))
            loss_history.append(epoch_loss)

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        return loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted risk probabilities for input feature matrix X."""
        X = np.asarray(X, dtype=float)
        probabilities, _ = self.forward(X)
        return probabilities

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary risk predictions (0 or 1) based on probability threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_parameters(self) -> dict:
        """
        Return a deep copy of model parameters (weights and biases).
        Prevents external mutation of internal network state.
        """
        return {key: val.copy() for key, val in self.parameters.items()}

    def set_parameters(self, new_params: dict) -> None:
        """
        Load parameters into the model after verifying keys and matrix shapes.

        Raises:
            ValueError: If parameter keys are missing or shapes do not match.
        """
        expected_keys = {"W1", "b1", "W2", "b2"}
        actual_keys = set(new_params.keys())
        if actual_keys != expected_keys:
            raise ValueError(f"Parameter key mismatch. Expected: {expected_keys}, Found: {actual_keys}")

        expected_shapes = {
            "W1": (self.input_dim, self.hidden_dim),
            "b1": (1, self.hidden_dim),
            "W2": (self.hidden_dim, self.output_dim),
            "b2": (1, self.output_dim)
        }

        for key, expected_shape in expected_shapes.items():
            param_array = np.asarray(new_params[key], dtype=float)
            if param_array.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{key}'. "
                    f"Expected shape {expected_shape}, got {param_array.shape}."
                )
            self.parameters[key] = param_array.copy()


def compute_proximal_loss(parameters: dict, global_parameters: dict, mu: float = 0.0) -> float:
    """
    Calculate FedProx proximal regularization penalty:
        (mu / 2) * sum_{k} ||w_k - w_{t, k}||^2

    Parameters:
        parameters (dict): Current local model parameters.
        global_parameters (dict): Initial global model parameters at start of round.
        mu (float): Proximal regularization coefficient (mu >= 0).

    Returns:
        float: Proximal penalty value.
    """
    return HealthcareMLP.compute_proximal_loss(parameters, global_parameters, mu=mu)


if __name__ == "__main__":
    print("Testing HealthcareMLP Architecture...")
    model = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=42)
    params = model.get_parameters()
    print("Initialized Parameter Shapes:")
    for k, v in params.items():
        print(f"  {k}: {v.shape}")

    # Dummy forward pass
    dummy_X = np.random.randn(10, 5)
    dummy_y = np.random.randint(0, 2, size=(10, 1))
    probs = model.predict_proba(dummy_X)
    loss = model.compute_loss(dummy_y, probs)
    print(f"Dummy Forward Pass Initial Loss: {loss:.4f}")
    print(f"Probabilities shape: {probs.shape}, range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("HealthcareMLP ready.")
