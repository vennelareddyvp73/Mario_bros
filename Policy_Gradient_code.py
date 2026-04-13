import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

class AC(nn.Module):
    def __init__(self, n_actions, input_channels=3):
        super(AC, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 210, 160)
            conv_out = self._forward_conv(dummy)
            n_flatten = conv_out.view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, 512)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def _forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = x / 255.0
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.actor(x), self.critic(x)

class MarioAgent:
    def __init__(self, n_actions, lr=2.5e-4, gamma=0.99, vf_coef=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AC(n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.n_actions = n_actions

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2,0,1).unsqueeze(0)
        logits, value = self.model(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in reversed(list(zip(rewards, dones))):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs, values, returns):
        states_t = torch.stack([
            torch.tensor(s, dtype=torch.float32, device=self.device).permute(2,0,1)
            for s in states
        ])
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)

        logits, new_values = self.model(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_t)
        advantages = returns_t - values_t

        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = nn.MSELoss()(new_values.squeeze(), returns_t)
        loss = policy_loss + self.vf_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(num_episodes=500, steps_per_ep=1000, save_path="mario_no_entropy.pth"):
    env = gym.make("ALE/MarioBros-v5", render_mode="rgb_array")
    agent = MarioAgent(n_actions=env.action_space.n)
    reward_log, value_log, step_log = [], [], []
    reward_buf, value_buf = deque(maxlen=25), deque(maxlen=25)

    for episode in tqdm(range(1, num_episodes + 1), desc="Training :"):
        state, _ = env.reset()
        ep_reward, ep_steps = 0, 0
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        for step in range(steps_per_ep):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, trunc, _ = env.step(action)
            shaped_reward = 0.1
            if reward > 0:
                shaped_reward += reward
            elif done:
                shaped_reward -= 1.0
            states.append(state)
            log_probs.append(log_prob.item())
            actions.append(action)
            values.append(value.item())
            rewards.append(shaped_reward)
            dones.append(done)
            ep_reward += shaped_reward
            ep_steps += 1
            state = next_state
            if done:
                break

        with torch.no_grad():
            last_value = agent.model(
                torch.tensor(state, dtype=torch.float32, device=agent.device).permute(2,0,1).unsqueeze(0)
            )[1].item()

        returns = agent.compute_returns(rewards, dones, last_value)
        agent.update(states, actions, log_probs, values, returns)
        reward_buf.append(ep_reward)
        value_buf.append(np.mean(values) if len(values) > 0 else 0.0)
        reward_log.append(np.mean(reward_buf))
        value_log.append(np.mean(value_buf))
        step_log.append(ep_steps)

        if episode % 25 == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {np.mean(reward_buf):.2f} | Avg Value: {np.mean(value_buf):.3f} | Steps: {ep_steps}")

    torch.save(agent.model.state_dict(), save_path)
    env.close()
    plot_learning_curves(reward_log, value_log, step_log)

def plot_learning_curves(reward_log, value_log, step_log):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(reward_log); plt.title("Reward Progress")
    plt.subplot(1, 3, 2)
    plt.plot(value_log); plt.title("Value Function")
    plt.subplot(1, 3, 3)
    plt.plot(step_log); plt.title("Steps per Episode")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train(num_episodes=500, steps_per_ep=1000)
