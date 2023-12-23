import sys
import gym
import torch
import pylab
import random
import numpy as np
import time
import os
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
import gymnasium
import configargparse
import importlib
import yaml
import argparse

"""
env_name: "SpaceInvaders-v0"
gym_factory: gym
num_envs: 8
model: "PPO_MHDPA_160"
evaluation_reward_length: 500
training_frame_count: 20000000
history_size: 8
graph_update_interval: 50
save_graph_folder: "./save_graph"
save_model_folder: "./save_model"
"""

def train_agent(env_name,
                gym_factory,
                num_envs,
                model,
                resize_shape,
                evaluation_reward_length,
                training_frame_count,
                history_size,
                graph_update_interval,
                save_graph_folder,
                save_model_folder,
                c1=1.0,
                c2=0.01):

    evaluation_reward = deque(maxlen=evaluation_reward_length)
    frame = 0
    memory_size = 0
    reset_max = 10

    print("\n\n\n ------- STARTING TRAINING FOR %s ------- \n\n\n" % (env_name))

    vis_env_idx = 0
    envs = []
    for i in range(num_envs):
        if i == vis_env_idx:
            render_mode="human"
        else:
            render_mode="rgb_array"
        envs.append(GameEnv(env_name,
                            render_mode=render_mode,
                            gym_factory=gym_factory,
                            history_size=history_size,
                            resize_shape=resize_shape))
    #env.render()


    number_lives = envs[0].life
    state_size = envs[0].observation_space.shape
    action_size = envs[0].action_space.n
    rewards, episodes = [], []

    print("Determing min/max rewards of environment")
    [low, high] = score_range = get_score_range(env_name, gym_factory=gym_factory)
    print("Min: %d. Max: %d." % (low, high))

    vis_env = envs[vis_env_idx]
    e = 0
    frame = 0
    max_eval = -np.inf
    reset_count = 0

    env_name = os.path.basename(env_name)

    agent = Agent(action_size,
                  mode=model,
                  c1=c1,
                  c2=c2)
    agent.policy_net.eval()
    torch.save(agent.policy_net.state_dict(),
               os.path.join(
                   save_model_folder,
                   os.path.basename(env_name) + "_best"
               ))

    evaluation_reward = deque(maxlen=evaluation_reward_length)
    frame = 0
    memory_size = 0
    reset_max = 10

    while (frame < training_frame_count):
        step = 0
        assert(num_envs * env_mem_size == train_frame)
        frame_next_vals = []

        for j in range(env_mem_size):

            curr_states = np.stack([envs[i].history[history_size-1,:,:] for i in range(num_envs)])
            next_states = []
            net_in = np.stack([envs[i].history[:history_size,:,:] for i in range(num_envs)])
            step += num_envs
            frame += num_envs

            action_availabilities = [envs[i].get_available_actions() for i in range(num_envs)]

            actions, values, _ = agent.get_action(np.float32(net_in) / 255.,
                                                  action_availabilities=action_availabilities)

            for i in range(num_envs):
                env = envs[i]
                next_state, env.reward, env.done, env.info = env.step(actions[i])
                next_states.append(next_state)
                if (i == vis_env_idx):
                    vis_env._env.render()

            for i in range(num_envs):
                env = envs[i]
                """
                next_state, env.reward, env.done, env.info = env.step(actions[i])
                if (i == vis_env_idx):
                    vis_env._env.render()
                """

                frame_next_state = get_frame(next_states[i])
                env.history[history_size,:,:] = frame_next_state
                terminal_state = env.done or check_live(env.life, env.info['lives'])
                env.life = env.info['lives']
                r = (env.reward / high) #np.log(max(env.reward+1, 1))#((env.reward - low) / (high - low)) * 30
                agent.memory.push(i, deepcopy(curr_states[i]), actions[i], r, terminal_state, values[i], 0, 0)

                env.score += env.reward
                env.history[:history_size, :, :] = env.history[1:,:,:]

                if (env.done):
                    if (e % 50 == 0):
                        print('now time : ', datetime.now())
                        rewards.append(np.mean(evaluation_reward))
                        episodes.append(e)
                        pylab.plot(episodes, rewards, 'b')
                        pylab.savefig(os.path.join(save_graph_folder,
                                                   f"{env_name}_ppo.png"))
                        torch.save(agent.policy_net,
                                   os.path.join(save_model_folder,
                                                f"{env_name}_ppo"))
                        #torch.save(agent.policy_net, "./save_model/" + name + "_ppo")

                        if np.mean(evaluation_reward) > max_eval:
                            torch.save(agent.policy_net,
                                       os.path.join(save_model_folder,
                                                    f"{env_name}_ppo_best"))
                            max_eval = float(np.mean(evaluation_reward))
                            reset_count = 0
                        elif e > 5000:
                            reset_count += 1
                            """
                            if (reset_count == reset_max):
                                print("Training went nowhere, starting again at best model")
                                agent.policy_net.load_state_dict(torch.load("./save_model/spaceinvaders_ppo_best"))
                                agent.update_target_net()
                                reset_count = 0
                            """
                    e += 1
                    evaluation_reward.append(env.score)
                    print("episode:", e, "  score:", env.score,  " epsilon:", agent.epsilon, "   steps:", step,
                      " evaluation reward:", np.mean(evaluation_reward))

                    env.done = False
                    env.score = 0
                    #env.history = np.zeros([HISTORY_SIZE+1,HEIGHT,WIDTH], dtype=np.uint8)
                    env.state = env.reset()
                    env.life = number_lives
                    get_init_state(env.history, env.state, height=env.height, width=env.width)

            if (j == env_mem_size-1):
                net_in = np.stack([envs[k].history[1:,:,:] for k in range(num_envs)])
                action_availabilities = [envs[i].get_available_actions() for i in range(num_envs)]
                _, frame_next_vals, _ = agent.get_action(np.float32(net_in) / 255.,
                                                         action_availabilities=action_availabilities)

        agent.train_policy_net(frame, frame_next_vals)
        agent.update_target_net()
    print("FINISHED TRAINING FOR %s" % (env_name))
    pylab.figure()

    for i in range(len(envs)):
        envs[i]._env.close()
    del envs

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_config")

    return parser.parse_args()

def main():

    args = parse_args()

    train_config = args.train_config
    with open(train_config, "r") as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    env_name = train_config["env_name"]
    gym_factory = train_config["gym_factory"]
    num_envs = train_config["num_envs"]
    model = train_config["model"]
    resize_shape = train_config["resize_shape"]
    evaluation_reward_length = train_config["evaluation_reward_length"]
    training_frame_count = train_config["training_frame_count"]
    history_size = train_config["history_size"]
    graph_update_interval = train_config["graph_update_interval"]
    save_graph_folder = train_config["save_graph_folder"]
    save_model_folder = train_config["save_model_folder"]
    c1 = train_config.get("c1", 1.0)
    c2 = train_config.get("c2", 0.01)

    gym_factory = importlib.import_module(gym_factory)

    train_agent(
        env_name,
        gym_factory,
        num_envs,
        model,
        resize_shape,
        evaluation_reward_length,
        training_frame_count,
        history_size,
        graph_update_interval,
        save_graph_folder,
        save_model_folder,
        c1=c1,
        c2=c2
    )

if __name__ == "__main__":
    main()
