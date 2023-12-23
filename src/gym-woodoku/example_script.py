import gym_woodoku
import gymnasium as gym

env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder')

observation, info = env.reset()
for i in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        env.reset()
env.close()
