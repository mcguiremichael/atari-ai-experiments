import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from config import *
import gym
import random

def find_max_lifes(env):
    env.reset()
    _, _, _, _, info = env.step(0)
    return info['lives']

def check_live(life, cur_life):
    if life > cur_life:
        return True
    else:
        return False

def get_frame(X,
              height=HEIGHT,
              width=WIDTH):

    if X.ndim == 2:
        return np.uint8(X * 255)

    x = np.uint8(resize(rgb2gray(X), (height, width), mode='reflect') * 255)
    return x

def get_init_state(history,
                   s,
                   height=HEIGHT,
                   width=WIDTH):
    for i in range(HISTORY_SIZE):
        history[i, :, :] = get_frame(s, height=height, width=width)

def get_score_range(name,
                    gym_factory=gym):
    env = gym_factory.make(name, render_mode="rgb_array")
    n = env.action_space.n
    max_score = -np.inf
    min_score = np.inf
    for i in range(15):
        done = False
        env.reset()
        while not done:
            action = random.randint(0, n-1)
            _, reward, done, _, _ = env.step(action)
            if (reward > max_score):
                max_score = reward
            if (reward < min_score):
                min_score = reward
    return [min_score, max_score]
