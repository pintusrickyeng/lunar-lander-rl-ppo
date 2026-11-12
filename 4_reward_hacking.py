import gymnasium as gym
from stable_baselines3 import PPO

class HoverWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        tilt = obs[4]
        
        if abs(tilt) < 0.10:
            reward += 1.0
            
        return obs, reward, terminated, truncated, info

print("Creazione ambiente con Reward Hacking...")
env = HoverWrapper(gym.make("LunarLander-v3"))

model = PPO("MlpPolicy", env, verbose=1)

print("Inizio addestramento (il razzo imparerà a fluttuare per rubare punti)...")
model.learn(total_timesteps=500_000)

model.save("ppo_lunar_hacker")
print("Modello 'hacker' salvato!")
env.close()