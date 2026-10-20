"""
Inference Engine for Live Smart Grid Dispatch Controller
========================================================
Supports both PyTorch and Pure-NumPy forward passes for greedy DQN action selection.
NumPy engine enables ultra-fast, lightweight execution on Raspberry Pi with zero torch overhead.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("smart_grid.inference")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NumpyQNetwork:
    """Pure NumPy implementation of the Q-Network."""

    def __init__(
        self,
        w0: np.ndarray, b0: np.ndarray,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
    ):
        self.w0, self.b0 = w0, b0
        self.w1, self.b1 = w1, b1
        self.w2, self.b2 = w2, b2
        self.w3, self.b3 = w3, b3

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass for (12,) or (batch, 12).
        Returns Q-values of shape (40,) or (batch, 40).
        """
        x = np.asarray(state, dtype=np.float32)
        # Linear 0 (12 -> 256) + ReLU
        h1 = np.dot(x, self.w0.T) + self.b0
        h1 = np.maximum(0.0, h1)

        # Linear 2 (256 -> 256) + ReLU
        h2 = np.dot(h1, self.w1.T) + self.b1
        h2 = np.maximum(0.0, h2)

        # Linear 4 (256 -> 128) + ReLU
        h3 = np.dot(h2, self.w2.T) + self.b2
        h3 = np.maximum(0.0, h3)

        # Linear 6 (128 -> 40)
        q_values = np.dot(h3, self.w3.T) + self.b3
        return q_values

    def select_action(self, state: np.ndarray) -> int:
        q = self.forward(state)
        return int(np.argmax(q, axis=-1))

    def save_weights(self, path: Union[str, Path]):
        np.savez(
            path,
            w0=self.w0, b0=self.b0,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
        )

    @classmethod
    def load_weights(cls, path: Union[str, Path]) -> "NumpyQNetwork":
        data = np.load(path)
        return cls(
            w0=data["w0"], b0=data["b0"],
            w1=data["w1"], b1=data["b1"],
            w2=data["w2"], b2=data["b2"],
            w3=data["w3"], b3=data["b3"],
        )


class DQNInferenceEngine:
    """
    Unified inference engine for live greedy forward passes (epsilon = 0).
    Loads checkpoint from file and configures either PyTorch or NumPy backend.
    """

    def __init__(self, model_path: Union[str, Path], engine_type: str = "torch"):
        self.model_path = Path(model_path)
        self.engine_type = engine_type.lower()
        self.torch_model = None
        self.numpy_net: Optional[NumpyQNetwork] = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self._load_model()

    def _load_model(self):
        if self.model_path.suffix == ".npz":
            logger.info(f"Loading NumPy weights from {self.model_path}")
            self.numpy_net = NumpyQNetwork.load_weights(self.model_path)
            self.engine_type = "numpy"
            return

        # Loading .pth checkpoint
        if not HAS_TORCH:
            raise RuntimeError(
                f"PyTorch is not installed to load {self.model_path}. "
                "Please export to .npz or install torch."
            )

        logger.info(f"Loading PyTorch checkpoint from {self.model_path} on CPU...")
        checkpoint = torch.load(self.model_path, map_location="cpu")

        # Extract state dict
        if "policy_net" in checkpoint:
            state_dict = checkpoint["policy_net"]
        elif "q_network" in checkpoint:
            state_dict = checkpoint["q_network"]
        else:
            state_dict = checkpoint

        # Check weights shapes to build Numpy network and Torch network
        w0 = state_dict["network.0.weight"].cpu().numpy()
        b0 = state_dict["network.0.bias"].cpu().numpy()
        w1 = state_dict["network.2.weight"].cpu().numpy()
        b1 = state_dict["network.2.bias"].cpu().numpy()
        w2 = state_dict["network.4.weight"].cpu().numpy()
        b2 = state_dict["network.4.bias"].cpu().numpy()
        w3 = state_dict["network.6.weight"].cpu().numpy()
        b3 = state_dict["network.6.bias"].cpu().numpy()

        self.numpy_net = NumpyQNetwork(w0, b0, w1, b1, w2, b2, w3, b3)

        # PyTorch model
        state_size = w0.shape[1]
        action_size = w3.shape[0]

        from agents.dqn_agent import QNetwork
        self.torch_model = QNetwork(state_size=state_size, action_size=action_size)
        self.torch_model.load_state_dict(state_dict)
        self.torch_model.eval()

        logger.info(
            f"Successfully loaded DQN (state_dim={state_size}, action_dim={action_size}, engine={self.engine_type})"
        )

    def predict(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Runs greedy deterministic forward pass.
        Returns:
            (selected_action: int, q_values: np.ndarray)
        """
        if self.engine_type == "numpy" or not HAS_TORCH:
            q_vals = self.numpy_net.forward(state)
            action = int(np.argmax(q_vals))
            return action, q_vals

        # PyTorch greedy pass (matches evaluate_dqn.py)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device="cpu"
        ).unsqueeze(0)

        with torch.no_grad():
            q_tensor = self.torch_model(state_tensor)
            q_vals = q_tensor.squeeze(0).numpy()
            action = int(q_tensor.argmax(dim=1).item())

        return action, q_vals

    def verify_numerical_parity(self, sample_state: Optional[np.ndarray] = None, atol: float = 1e-4) -> bool:
        """Verifies that PyTorch and NumPy forward passes match within tolerance."""
        if self.torch_model is None or self.numpy_net is None:
            return True

        if sample_state is None:
            sample_state = np.random.randn(12).astype(np.float32)

        _, q_np = DQNInferenceEngine.predict(self, sample_state)  # will use configured engine
        # Test both
        state_tensor = torch.tensor(sample_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_torch = self.torch_model(state_tensor).squeeze(0).numpy()
        q_numpy = self.numpy_net.forward(sample_state)

        diff = np.max(np.abs(q_torch - q_numpy))
        is_close = diff < atol
        if not is_close:
            logger.error(f"Inference engine parity mismatch! max diff: {diff}")
        else:
            logger.debug(f"Inference engine parity verified (max diff: {diff:.8f})")
        return bool(is_close)
