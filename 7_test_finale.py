import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo

model = PPO.load("ppo_lunar_fixed")

print("Valutazione del modello su 10 tentativi (senza wrapper)...")
eval_env = gym.make("LunarLander-v3")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"\n--- RISULTATO FINALE ---")
print(f"Punteggio medio: {mean_reward:.2f} +/- {std_reward:.2f}")

if mean_reward >= 200:
    print("VITTORIA! Il lander ha superato i 200 punti ed è addestrato perfettamente.")
else:
    print("Punteggio sotto i 200. (Essendo un processo stocastico, a volte serve solo riaddestrarlo o aumentare i frame).")
eval_env.close()

print("\nRegistrazione del video dell'atterraggio...")
vid_env = gym.make("LunarLander-v3", render_mode="rgb_array")
vid_env = RecordVideo(vid_env, video_folder="video_atterraggio", name_prefix="lander_fixed")

obs, info = vid_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vid_env.step(action)
    done = terminated or truncated

vid_env.close()
print("Video salvato nella cartella 'video_atterraggio'!")