"""
Deep Q-Network (DQN) implementation with experience replay and target network.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """
    Multi-layer fully-connected DQN with LayerNorm for stable training.

    Architecture:
        Input(state_dim) → [Linear → LayerNorm → ReLU] × len(hidden_dims)
        → Linear(hidden_dims[-1], action_dim)
    """

    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 5,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    """Fixed-capacity circular replay buffer."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states      = torch.tensor(np.stack([t.state      for t in batch]), dtype=torch.float32, device=device)
        actions     = torch.tensor([t.action    for t in batch], dtype=torch.long,    device=device)
        rewards     = torch.tensor([t.reward    for t in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones       = torch.tensor([t.done      for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent with:
    - Epsilon-greedy exploration
    - Experience replay
    - Target network with periodic hard updates
    - Huber loss (SmoothL1) for robustness to outlier rewards
    """

    def __init__(
        self,
        state_dim:          int   = 24,
        action_dim:         int   = 5,
        hidden_dims:        list[int] | None = None,
        lr:                 float = 1e-3,
        gamma:              float = 0.99,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        epsilon_decay:      float = 0.995,
        batch_size:         int   = 64,
        target_update_freq: int   = 100,
        buffer_capacity:    int   = 10_000,
        device:             str   = "auto",
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy parameters
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.loss_fn      = nn.SmoothL1Loss()

        self.steps_done = 0

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        """
        Sample a mini-batch, compute TD targets, backpropagate.

        Returns the scalar loss value, or None if the buffer is not yet
        large enough for a full batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q-values for chosen actions
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD targets using target network
        with torch.no_grad():
            q_next    = self.target_net(next_states).max(dim=1).values
            q_targets = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        self._update_epsilon()

        # Hard-copy policy → target every N steps
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def _update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_net":    self.policy_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "steps_done":    self.steps_done,
            "state_dim":     self.state_dim,
            "action_dim":    self.action_dim,
            "gamma":         self.gamma,
            "epsilon_end":   self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["policy_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon    = checkpoint.get("epsilon",    self.epsilon_end)
        self.steps_done = checkpoint.get("steps_done", 0)
        self.target_net.eval()
