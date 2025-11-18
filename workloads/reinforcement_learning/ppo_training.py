#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Training Workload
Trains RL agents using PPO on synthetic environments.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    load_config,
    get_logger,
    PerformanceBenchmark,
    GPUMonitor,
    SyntheticReinforcementEnvironment
)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, state_dim: int, action_dim: int, continuous: bool = True):
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            continuous: Whether actions are continuous
        """
        super().__init__()
        self.continuous = continuous

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(256, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(256, action_dim)

        # Critic head
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        """Forward pass."""
        features = self.shared(state)
        value = self.critic(features)

        if self.continuous:
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)
            return action_mean, action_std, value
        else:
            action_logits = self.actor(features)
            return action_logits, value

    def get_action(self, state):
        """Sample action from policy."""
        if self.continuous:
            action_mean, action_std, value = self.forward(state)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob, value
        else:
            action_logits, value = self.forward(state)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, value


class PPOTrainer:
    """PPO trainer."""

    def __init__(self, config_path: str = None):
        """Initialize PPO trainer."""
        self.config = load_config(config_path)

        self.logger = get_logger(
            name="ppo_training",
            log_dir=self.config.get('general.output_dir', './results'),
            level=self.config.get('general.log_level', 'INFO')
        )

        self.device = torch.device(
            self.config.get('gpu.device', 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )

        self.gpu_monitor = GPUMonitor(device_ids=self.config.get('gpu.device_ids', [0]))

        self.benchmark = PerformanceBenchmark(
            name="ppo_training",
            output_dir=self.config.get('general.output_dir', './results')
        )

        # PPO hyperparameters
        self.num_steps = self.config.get('workloads.reinforcement_learning.ppo.num_steps', 2048)
        self.num_epochs = self.config.get('workloads.reinforcement_learning.ppo.num_epochs', 10)
        self.mini_batch_size = self.config.get('workloads.reinforcement_learning.ppo.mini_batch_size', 64)
        self.clip_epsilon = self.config.get('workloads.reinforcement_learning.ppo.clip_epsilon', 0.2)
        self.lr = self.config.get('training.learning_rate', 0.0003)
        self.total_episodes = self.config.get('training.epochs', 100)

        self.logger.log_header("PPO Training")
        self._log_configuration()

    def _log_configuration(self):
        """Log configuration."""
        config_dict = {
            'Device': str(self.device),
            'Total Episodes': self.total_episodes,
            'Steps per Episode': self.num_steps,
            'PPO Epochs': self.num_epochs,
            'Mini Batch Size': self.mini_batch_size,
            'Clip Epsilon': self.clip_epsilon,
            'Learning Rate': self.lr,
        }
        self.logger.log_config(config_dict)

    def create_env_and_model(self):
        """Create environment and model."""
        self.logger.info("Creating synthetic RL environment...")

        env = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=2,
            continuous_action=True,
            episode_length=200,
            seed=self.config.get('general.seed', 42)
        )

        model = ActorCritic(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            continuous=env.continuous_action
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model parameters: {total_params:,}")

        return env, model

    def collect_trajectories(self, env, model):
        """Collect trajectories."""
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        state = env.reset()
        episode_reward = 0

        for _ in range(self.num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = model.get_action(state_tensor)

            action_np = action.cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action_np)

            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                state = env.reset()

        return {
            'states': torch.FloatTensor(states).to(self.device),
            'actions': torch.FloatTensor(actions).to(self.device),
            'log_probs': torch.FloatTensor(log_probs).to(self.device),
            'rewards': rewards,
            'values': values,
            'dones': dones
        }

    def compute_returns(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """Compute returns and advantages."""
        returns = []
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def train(self):
        """Run PPO training."""
        self.logger.log_header("Starting PPO Training")

        env, model = self.create_env_and_model()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for episode in range(1, self.total_episodes + 1):
            self.benchmark.start_iteration()

            # Collect trajectories
            trajectory = self.collect_trajectories(env, model)

            # Compute returns
            returns, advantages = self.compute_returns(
                trajectory['rewards'],
                trajectory['values'],
                trajectory['dones']
            )

            # PPO update
            for _ in range(self.num_epochs):
                action_mean, action_std, values = model(trajectory['states'])
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(trajectory['actions']).sum(-1)

                ratio = torch.exp(new_log_probs - trajectory['log_probs'])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(), returns)
                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            iter_time = self.benchmark.end_iteration()
            avg_reward = np.mean(trajectory['rewards'])

            self.benchmark.record_loss(loss.item())

            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}/{self.total_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Loss: {loss.item():.4f}"
                )

        self.logger.log_header("PPO Training Complete")
        self.benchmark.print_summary()
        self.benchmark.save_results()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PPO Training Workload')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    args = parser.parse_args()

    trainer = PPOTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
