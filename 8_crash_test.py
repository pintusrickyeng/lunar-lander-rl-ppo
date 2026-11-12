import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="video_atterraggio", name_prefix="lander_crash")

obs, info = env.reset()
done = False
punteggio_totale = 0

print("Avvio simulazione casuale (preparati allo schianto)...")

while not done:
    
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    punteggio_totale += reward
    done = terminated or truncated

env.close()
print(f"Schianto completato! Punteggio finale disastroso: {punteggio_totale:.2f}")
print("Video salvato nella cartella 'video_atterraggio' con il nome 'lander_crash'.")