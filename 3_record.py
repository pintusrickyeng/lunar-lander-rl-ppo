import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

env = gym.make("LunarLander-v3", render_mode="rgb_array")

env = RecordVideo(env, video_folder="video_atterraggio", name_prefix="lander_baseline")


model = PPO.load("ppo_lunar_baseline")


obs, info = env.reset()
done = False
punteggio_totale = 0

print("Inizio simulazione e registrazione del video...")

while not done:

    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    punteggio_totale += reward
    done = terminated or truncated

env.close()
print(f"Atterraggio completato con un punteggio di: {punteggio_totale:.2f}")
print("Video salvato nella cartella 'video_atterraggio'!")