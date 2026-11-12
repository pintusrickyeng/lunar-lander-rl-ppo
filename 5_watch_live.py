import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load("ppo_lunar_hacker")
obs, info = env.reset()
done = False

print("Inizio simulazione live...")

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()