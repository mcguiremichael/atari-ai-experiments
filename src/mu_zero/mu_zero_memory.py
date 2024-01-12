
from collections import deque

import torch.utils.data as data
import numpy as np
import random
from typing import Optional, Tuple

class ReplayMemory(data.Dataset):

    def __init__(self,
                 memory_length : int,
                 history_size : int,
                 search_depth : int):

        self.memory = deque(maxlen=memory_length)
        self.history_size = history_size
        self.search_depth = search_depth

    def push(self,
             state,
             action,
             reward,
             terminal):

        self.memory.append([state, action, reward, terminal])

    def __getitem__(self, index : int) -> Tuple[
        np.array,
        np.array,
        np.array,
        np.array
    ]:

        curr_state, curr_action, curr_reward, curr_terminal = self.memory[index]

        history_states = []
        history_actions = []

        terminal_encountered_in_history = False

        for i in range(1, self.history_size):

            adjusted_index = index - i

            if adjusted_index < 0:
                past_state = np.zeros_like(curr_state)
                past_action = 0

                history_states.append(past_state)
                history_actions.append(past_action)
                continue

            past_state, past_action, _, past_terminal = self.memory[adjusted_index]

            if past_terminal:
                terminal_encountered_in_history = True

            if terminal_encountered_in_history:
                past_state = np.zeros_like(curr_state)
                past_action = 0

            history_states.append(past_state)
            history_actions.append(past_action)

        adjusted_index -= 1
        if adjusted_index < 0 or self.memory[adjusted_index][3]:
            history_actions.append(0)
        else:
            history_actions.append(self.memory[adjusted_index][1])

        history_states = history_states[::-1]
        history_actions = history_actions[::-1]

        future_states = []
        future_actions = []
        future_rewards = []
        future_terminals = []

        terminal_encountered_in_future = False

        for i in range(1, self.search_depth):

            adjusted_index = index + i

            #assert(adjusted_index < len(self))

            future_state, future_action, future_reward, future_terminal = self.memory[adjusted_index]

            if terminal_encountered_in_future:
                future_state = np.zeros_like(curr_state)
                future_action = 0
                future_reward = 0
                future_terminal = True

            future_states.append(future_state)
            future_actions.append(future_action)
            future_rewards.append(future_reward)
            future_terminals.append(future_terminal)

            if future_terminal:
                terminal_encountered_in_future = True

        states = np.stack(history_states + [curr_state] + future_states, axis=0)
        actions = np.array(history_actions + [curr_action] + future_actions)
        rewards = np.array([curr_reward] + future_rewards)
        terminals = np.array([curr_terminal] + future_terminals).astype(bool)

        return states, actions, rewards, terminals

    def __len__(self) -> int:

        length = len(self.memory) - self.search_depth

        if length < 0:
            return 0

        return length

class MultiEnvReplayMemory(data.Dataset):

    def __init__(self,
                 num_envs : int,
                 per_env_memory_length : int,
                 history_size : int,
                 search_depth : int):

        self.memory_per_env = [ReplayMemory(
            per_env_memory_length,
            history_size,
            search_depth
        ) for i in range(num_envs)]

    def push(self,
             index : int,
             state,
             action,
             reward,
             terminal):

        self.memory_per_env[index].push(
            state,
            action,
            reward,
            terminal
        )

    def sample_mini_batch(self, batch_size : int):

        indices = np.random.randint(0, len(self), batch_size)

        states, actions, rewards, terminals = list(zip(*[
            self[i] for i in indices
        ]))

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)
        terminals = np.stack(terminals, axis=0)

        return states, actions, rewards, terminals

    def __getitem__(self, index : int):

        lengths = [len(x) for x in self.memory_per_env]
        cumulative_lengths = np.cumsum(lengths)
        env_idx = np.searchsorted(cumulative_lengths, index)
        sub_index = index % lengths[0]

        return self.memory_per_env[env_idx][sub_index]

    def __len__(self) -> int:
        return sum([len(x) for x in self.memory_per_env])
