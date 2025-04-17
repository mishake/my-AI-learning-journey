import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=5000)

# Run the trained agent
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
env.close()
