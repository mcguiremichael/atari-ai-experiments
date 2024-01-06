
from utils import *
from agent import *
from model import *
from config import *

import numpy as np

from copy import deepcopy

def generate_gameplay_data(agent,
                           envs,
                           history_size : int,
                           vis_env_idx : int):
    
    num_envs = len(envs)
    vis_env = envs[vis_env_idx]
    number_lives = envs[0].life

    prev_actions = None

    while True:

        curr_states = np.stack([envs[i].history[history_size-1,:,:] for i in range(len(envs))])
        next_states = []
        net_in = np.stack([envs[i].history[:history_size,:,:] for i in range(num_envs)])
        curr_actions = np.stack([envs[i].action_history[:history_size] for i in range(num_envs)])

        action_availabilities = [envs[i].get_available_actions() for i in range(num_envs)]

        actions = agent.get_action(np.float32(net_in) / 255.,
                                   curr_actions,
                                   action_availabilities=action_availabilities)

        for i in range(num_envs):
            env = envs[i]
            next_state, env.reward, env.done, env.info = env.step(actions[i])
            next_states.append(next_state)
            if (i == vis_env_idx):
                vis_env._env.render()

        replay_memory_snapshots = []

        for i in range(num_envs):

            env = envs[i]
            frame_next_state = get_frame(next_states[i],
                                         envs[0].height,
                                         envs[0].width)
            env.history[history_size,:,:] = frame_next_state
            
            if prev_actions is not None:
                env.action_history[history_size] = prev_actions[i]
            
            terminal_state = env.done or check_live(env.life, env.info['lives'])
            env.life = env.info['lives']
            r = env.reward
            env.score += env.reward
            env.history[:history_size, :, :] = env.history[1:,:,:]
            env.action_history[:history_size] = env.action_history[1:]

            replay_memory_snapshots.append((
                i,
                deepcopy(curr_states[i]),
                actions[i],
                r,
                terminal_state
            ))

            if env.done:
                env.done = False
                env.score = 0
                env.state = env.reset()
                env.life = number_lives
                get_init_state(env.history, env.state, height=env.height, width=env.width)
                if prev_actions is not None:
                    prev_actions[i] = 0

        prev_actions = actions

        yield replay_memory_snapshots

