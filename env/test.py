import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/AirRaid-v5")
obs, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    obs, reward, teriminated, truncated, info = env.step(action)
    if not teriminated:
        obs1 = obs
    episode_over = teriminated or truncated
print('obs', obs1)
print('reward', reward)
print('action', action)
print('info', info)

env.close()

