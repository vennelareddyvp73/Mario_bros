# Super Mario Bros Reinforcement Learning

A deep reinforcement learning project implementing Actor-Critic and Proximal Policy Optimization (PPO) algorithms to train an agent to play Super Mario Bros using the Arcade Learning Environment.

## Project Overview

This project trains an AI agent to play Super Mario Bros using policy gradient methods. Two implementations are included:

1. Basic Actor-Critic (Policy Gradient)
2. Proximal Policy Optimization (PPO)

## Technical Architecture

### Neural Network Design

The agent uses a convolutional neural network with the following architecture:

1. Conv2D Layer 1: 32 filters, 8x8 kernel, stride 4
2. Conv2D Layer 2: 64 filters, 4x4 kernel, stride 2
3. Conv2D Layer 3: 64 filters, 3x3 kernel, stride 1
4. Fully Connected: 512 units
5. Actor Head: Output layer for action probabilities
6. Critic Head: Single output for state value estimation

Input: 210x160x3 RGB frames from the game environment

### Reward Shaping

The training uses custom reward shaping to improve learning:

- Base reward: 0.1 per step (encourages exploration)
- Positive game reward: Added to base reward
- Terminal penalty: -1.0 for episode end without completion

## Training Results

### Policy Gradient (Actor-Critic)

Training Configuration:
- Episodes: 500
- Steps per episode: 1000 (max)
- Learning rate: 2.5e-4
- Gamma (discount factor): 0.99
- Value function coefficient: 0.5

Performance Metrics:
- Peak average reward: 530.74 (Episode 50)
- Final average reward: 121.40 (Episode 500)
- Average value function: Ranges from 9.36 to 50.99
- Episode completion: Variable (313-1000 steps)

Key Observations:
- High variance in reward progress
- Early peak performance around episode 50
- Unstable learning with significant fluctuations
- Value function shows high volatility

### Proximal Policy Optimization (PPO)

Training Configuration:
- Episodes: 500
- Steps per episode: 1000 (max)

Performance Metrics:
- Peak average reward: 442.30 (Episode 450)
- Episode 25: 262.93 avg reward
- Episode 100: 226.05 avg reward
- Episode 400: 432.18 avg reward
- Final average reward: 96.75 (Episode 500)

Key Observations:
- More stable learning compared to basic Policy Gradient
- Improved performance in later episodes (375-450)
- Better value function estimates (up to 77.60)
- Reduced variance in training progression

## Algorithm Comparison

### Policy Gradient Strengths
- Simpler implementation
- Faster per-episode training time (1.43-2.77s/it)
- Direct policy optimization

### Policy Gradient Weaknesses
- High variance in rewards
- Unstable learning curve
- Performance degradation over time
- Difficulty maintaining good policies

### PPO Strengths
- More stable training
- Better long-term performance
- Improved value estimates
- Consistent progress in later stages

### PPO Weaknesses
- Slightly slower training iterations (1.52-2.42s/it)
- More complex implementation
- Requires additional hyperparameters

## Key Findings

1. Learning Stability: PPO demonstrates more stable learning compared to vanilla Policy Gradient
2. Peak Performance: Policy Gradient achieved higher peak rewards (530.74) but failed to maintain them
3. Sustained Performance: PPO shows better sustained performance after episode 300
4. Value Estimation: PPO produces more reliable value function estimates
5. Episode Length: Both methods often reach maximum episode length (1000 steps)

## Dependencies

- gymnasium
- torch
- numpy
- matplotlib
- tqdm

## Installation

```bash
pip install gymnasium torch numpy matplotlib tqdm
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

## Usage

### Training the Policy Gradient Agent

```python
python Policy_Gradient_code.py
```

Default parameters:
- 500 training episodes
- 1000 steps per episode
- Model saved as "mario_no_entropy.pth"

### Custom Training

```python
train(num_episodes=1000, steps_per_ep=1500, save_path="custom_model.pth")
```

## Training Monitoring

The training process displays:
- Episode number and total episodes
- Average reward (rolling 25-episode window)
- Average value function estimate
- Steps taken in current episode
- Training progress bar with time estimates

## Output

Training produces:
1. Saved model weights (.pth file)
2. Three performance graphs:
   - Reward Progress
   - Value Function Progress
   - Steps per Episode



