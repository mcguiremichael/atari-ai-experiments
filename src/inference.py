import sys
import gym
import torch
import pylab
import random
import numpy as np
import time
from collections import deque
from datetime import datetime
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.backends.cudnn.benchmarks = True
from torch.autograd import Variable
from utils import *
from agent import *
from model import *
from config import *
from env import GameEnv
import cv2
import os

def run_agent_on_game_and_visualize_attention(env_name,
                                              checkpoint_path,
                                              agent_mode,
                                              vis_folder,
                                              atten_vis_folder):

    env = GameEnv(env_name)
    number_lives = env.life
    state_size = env.observation_space.shape
    if (env_name == 'SpaceInvaders-v0' or env_name == 'Breakout-v0'):
        action_size = 4
    else:
        action_size = env.action_space.n

    agent = Agent(action_size, mode='PPO_MHDPA')

    i = 0
    while not env.done:

        i += 1
        next_states = []

        curr_state = env.history[HISTORY_SIZE-1,:,:]
        net_in = np.expand_dims(env.history[:HISTORY_SIZE,:,:], 0)
        actions, values, _ = agent.get_action(np.float32(net_in) / 255.)

        next_state, env.reward, env.done, env.info = env.step(actions[0])
        next_states.append(next_state)
        env._env.render()

        frame_next_state = get_frame(next_states[0])
        env.history[HISTORY_SIZE,:,:] = frame_next_state
        terminal_state = env.done
        env.life = env.info['lives']
        r = env.reward
        agent.memory.push(
            0,
            deepcopy(curr_state),
            actions[0],
            r,
            terminal_state,
            values[0],
            0,
            0
        )

        env.score += env.reward
        env.history[:HISTORY_SIZE, :, :] = env.history[1:,:,:]

        attention_maps = agent.get_attention_maps(np.float32(net_in) / 255.)

        save_frame(curr_state, vis_folder, i)
        save_attention_map_images(attention_maps, atten_vis_folder, i)

        if env.done:
            return

def save_frame(state_frame, vis_folder, idx):
    os.makedirs(vis_folder, exist_ok=True)
    cv2.imwrite(os.path.join(vis_folder, "%06d.png" % (idx)),
                cv2.cvtColor(state_frame, cv2.COLOR_RGB2BGR))

def save_attention_map_images(attention_maps, attention_folder, idx):
    os.makedirs(attention_folder, exist_ok=True)
    curr_frame_attention_folder = os.path.join(attention_folder,
                                               "attention_%06d" % (idx))

    os.makedirs(curr_frame_attention_folder, exist_ok=True)

    for k in range(len(attention_maps)):
        attention_map = attention_maps[k]
        h, w = attention_map.shape[1:3]
        for i in range(h):
            for j in range(w):

                image_path = os.path.join(
                    curr_frame_attention_folder,
                    "image_%d_%d_%d.png" % (k+1,i+1,j+1)
                )

                image = attention_map[0][i][j].cpu().detach().numpy()
                print(image.min(), image.max())
                image = image - np.min(image)
                image = 255 * (image / np.max(image))
                image = cv2.resize(image, (1000,1000), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(
                    image_path,
                    image
                )


run_agent_on_game_and_visualize_attention(
    "SpaceInvaders-v0",
    "SpaceInvaders-v0_ppo",
    "PPO_MHDPA",
    "frame_vis",
    "attention_vis"
)
