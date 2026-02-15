import gymnasium as gym
import torch
import torch.nn as nn
import time

# Environment
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Network
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
policy_net.load_state_dict(torch.load("cartpole_dqn.pth"))
policy_net.eval()

state, _ = env.reset()
state = torch.FloatTensor(state)
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        action = torch.argmax(policy_net(state)).item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = torch.FloatTensor(next_state)
    total_reward += reward
    time.sleep(0.02)

print("Total reward:", total_reward)
env.close()
