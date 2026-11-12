import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class FixedShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_shaping = None

    def _calc_potential(self, obs):
        x, y, vx, vy, tilt = obs[0], obs[1], obs[2], obs[3], obs[4]
        # Restituisce un potenziale negativo (più è vicino a 0, meglio è)
        return -10.0 * abs(tilt) - 5.0 * np.hypot(vx, vy)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_shaping = self._calc_potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_shaping = self._calc_potential(obs)
        reward += (current_shaping - self.prev_shaping)
        self.prev_shaping = current_shaping
        
        reward -= 0.05
        
        return obs, reward, terminated, truncated, info

print("Creazione ambiente con Reward Corretto...")
env = FixedShapingWrapper(gym.make("LunarLander-v3"))

model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,        
    gae_lambda=0.98,
    ent_coef=0.01,      
    verbose=1
)

print("Inizio addestramento ottimizzato...")
model.learn(total_timesteps=2_000_000)

model.save("ppo_lunar_fixed")
print("Modello 'fixed' salvato!")
env.close()