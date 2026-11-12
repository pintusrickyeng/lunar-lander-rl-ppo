import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

model = PPO("MlpPolicy", env, verbose=1)

print("Inizio addestramento...")
model.learn(total_timesteps=1_000_000)

model.save("ppo_lunar_baseline")
print("Modello salvato con successo come 'ppo_lunar_baseline.zip'")

env.close()