# LunarLander-v3: Attitude Control via PPO

In this project, I wanted to explore how Reinforcement Learning can be used to stabilize and land a lunar module, bridging the gap between traditional control theory and AI. I used Proximal Policy Optimization (PPO) to train an agent to solve the LunarLander-v3 environment.

## The Problem: Falling for Reward Hacking
While testing the environment, I ran into a classic RL issue: reward hacking. 
I wrote a custom wrapper to reward the lander for keeping a perfectly vertical pitch. However, instead of learning to land safely, the neural network found a mathematical loophole. It realized it could just hover in mid-air indefinitely, farming points without ever touching the ground. It did exactly what I optimized it for, but not what I actually wanted.

## The Fix: Reward Shaping
To fix this behavior, I had to rethink the reward structure:
- Time Penalty: I added a small negative reward for every frame spent in the air. This forced the agent to descend to stop losing points.
- Potential-based Rewards: Instead of rewarding absolute positioning, I rewarded the difference in stability between frames.

After training for about 1.5 million steps with these new rules (and tuning the entropy and gamma hyperparameters), the agent finally learned to land smoothly and efficiently, consistently beating the 200-point solving threshold.

## Tech Stack
- Python
- Gymnasium
- Stable-Baselines3 (PPO)# LunarLander-v3: Attitude Control via PPO

In this project, I wanted to explore how Reinforcement Learning can be used to stabilize and land a lunar module, bridging the gap between traditional control theory and AI. I used Proximal Policy Optimization (PPO) to train an agent to solve the LunarLander-v3 environment.

## The Problem: Falling for Reward Hacking
While testing the environment, I ran into a classic RL issue: reward hacking. 
I wrote a custom wrapper to reward the lander for keeping a perfectly vertical pitch. However, instead of learning to land safely, the neural network found a mathematical loophole. It realized it could just hover in mid-air indefinitely, farming points without ever touching the ground. It did exactly what I optimized it for, but not what I actually wanted.

## The Fix: Reward Shaping
To fix this behavior, I had to rethink the reward structure:
- Time Penalty: I added a small negative reward for every frame spent in the air. This forced the agent to descend to stop losing points.
- Potential-based Rewards: Instead of rewarding absolute positioning, I rewarded the difference in stability between frames.

After training for about 1.5 million steps with these new rules (and tuning the entropy and gamma hyperparameters), the agent finally learned to land smoothly and efficiently, consistently beating the 200-point solving threshold.

## Tech Stack
- Python
- Gymnasium
- Stable-Baselines3 (PPO)
