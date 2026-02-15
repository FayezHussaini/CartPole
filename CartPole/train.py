import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n              # 2

# Neural Network
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Replay buffer
buffer = deque(maxlen=10000)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
episodes = 1000

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    done = False

    while not done:
        # Action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(policy_net(state)).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)

        buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Learning step
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.stack(next_states)
            dones = torch.BoolTensor(dones)

            current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = target_net(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (~dones)

            loss = loss_fn(current_q, target_q.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}")

# Save model
torch.save(policy_net.state_dict(), "cartpole_dqn.pth")
env.close()
print("Training finished. Model saved.")
