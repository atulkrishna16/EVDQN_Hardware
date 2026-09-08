import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):

    def __init__(
        self,
        state_size,
        action_size,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                state_size,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                action_size
            ),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0005,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=100000,
        target_update_frequency=250,
        seed=42,
    ):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.target_update_frequency = (
            target_update_frequency
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"DQN device: {self.device}"
        )

        self.policy_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(
            capacity=buffer_size
        )

        self.training_steps = 0

    # ========================================================
    # ACTION
    # ========================================================

    def select_action(
        self,
        state,
        training=True,
    ):

        if (
            training
            and random.random() < self.epsilon
        ):

            return random.randrange(
                self.action_size
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

        return int(
            q_values.argmax(
                dim=1
            ).item()
        )

    # ========================================================
    # STORE
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(
            self.batch_size
        )

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        # Current Q
        current_q = self.policy_net(
            states
        ).gather(
            1,
            actions
        )

        # Target Q
        with torch.no_grad():

            next_q = self.target_net(
                next_states
            ).max(
                dim=1,
                keepdim=True
            )[0]

            target_q = (
                rewards
                + (
                    1.0 - dones
                )
                * self.gamma
                * next_q
            )

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):

            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(
            loss.item()
        )

    # ========================================================
    # EPSILON
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, path):

        torch.save(
            {
                "policy_net":
                    self.policy_net.state_dict(),

                "target_net":
                    self.target_net.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "training_steps":
                    self.training_steps,

                "state_size":
                    self.state_size,

                "action_size":
                    self.action_size,
            },
            path
        )

        print(
            f"Model saved: {path}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon_min
        )

        self.training_steps = checkpoint.get(
            "training_steps",
            0
        )

        print(
            f"Model loaded: {path}"
        )