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

def train_agent():
    evaluation_reward = deque(maxlen=evaluation_reward_length)
    frame = 0
    memory_size = 0
    reset_max = 10


    ### Loop through all environments and run PPO on them

    env_names = ['SpaceInvaders-v0', 'Boxing-v0', 'DoubleDunk-v0', 'IceHockey-v0', 'Breakout-v0', 'Phoenix-v0', 'Asteroids-v0', 'MsPacman-v0', 'Asterix-v0', 'Atlantis-v0', 'Alien-v0', 'Amidar-v0', 'Assault-v0', 'BankHeist-v0']
    #env_names = ['SpaceInvaders-v4']
    for a in range(len(env_names)):
        name = env_names[a]
        print("\n\n\n ------- STARTING TRAINING FOR %s ------- \n\n\n" % (name))

        envs = []
        for i in range(num_envs):
            envs.append(GameEnv(name))
        #env.render()


        number_lives = envs[0].life
        state_size = envs[0].observation_space.shape
        if (name == 'SpaceInvaders-v0' or name == 'Breakout-v0'):
            action_size = 4
        else:
            action_size = envs[0].action_space.n
        rewards, episodes = [], []

        vis_env_idx = 0
        vis_env = envs[vis_env_idx]
        e = 0
        frame = 0
        max_eval = -np.inf
        reset_count = 0


        agent = Agent(action_size, mode='PPO_MHDPA')
        torch.save(agent.policy_net.state_dict(), "./save_model/" + name + "_best")
        evaluation_reward = deque(maxlen=evaluation_reward_length)
        frame = 0
        memory_size = 0
        reset_max = 10

        print("Determing min/max rewards of environment")
        [low, high] = score_range = get_score_range(name)
        print("Min: %d. Max: %d." % (low, high))

        while (frame < 20000000):
            step = 0
            assert(num_envs * env_mem_size == train_frame)
            frame_next_vals = []

            for j in range(env_mem_size):

                curr_states = np.stack([envs[i].history[HISTORY_SIZE-1,:,:] for i in range(num_envs)])
                next_states = []
                net_in = np.stack([envs[i].history[:HISTORY_SIZE,:,:] for i in range(num_envs)])
                step += num_envs
                frame += num_envs
                actions, values, _ = agent.get_action(np.float32(net_in) / 255.)

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
                    env.history[HISTORY_SIZE,:,:] = frame_next_state
                    terminal_state = env.done #check_live(env.life, env.info['ale.lives'])
                    env.life = env.info['ale.lives']
                    r = env.reward #(env.reward / high) * 20.0 #np.log(max(env.reward+1, 1))#((env.reward - low) / (high - low)) * 30
                    agent.memory.push(i, deepcopy(curr_states[i]), actions[i], r, terminal_state, values[i], 0, 0)

                    if (j == env_mem_size-1):
                        net_in = np.stack([envs[k].history[1:,:,:] for k in range(num_envs)])
                        _, frame_next_vals, _ = agent.get_action(np.float32(net_in) / 255.)

                    env.score += env.reward
                    env.history[:HISTORY_SIZE, :, :] = env.history[1:,:,:]

                    if (env.done):
                        if (e % 50 == 0):
                            print('now time : ', datetime.now())
                            rewards.append(np.mean(evaluation_reward))
                            episodes.append(e)
                            pylab.plot(episodes, rewards, 'b')
                            pylab.savefig("./save_graph/" + name + "_ppo.png")
                            torch.save(agent.policy_net, "./save_model/" + name + "_ppo")

                            if np.mean(evaluation_reward) > max_eval:
                                torch.save(agent.policy_net.state_dict(), "./save_model/"  + name + "_ppo_best")
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
                        env.history = np.zeros([HISTORY_SIZE+1,HEIGHT,WIDTH], dtype=np.uint8)
                        env.state = env.reset()
                        env.life = number_lives
                        get_init_state(env.history, env.state)

            agent.train_policy_net(frame, frame_next_vals)
            agent.update_target_net()
        print("FINISHED TRAINING FOR %s" % (name))
        pylab.figure()

        for i in range(len(envs)):
            envs[i]._env.close()
        del envs

def main():
    train_agent()

if __name__ == "__main__":
    main()
