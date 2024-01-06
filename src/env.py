import gymnasium
import gym
import numpy as np
from utils import *
from config import *
import gym_woodoku

class GameEnv():

    def __init__(self,
                 name,
                 gym_factory=gym,
                 history_size=HISTORY_SIZE,
                 resize_shape=(160, 160),
                 render_mode="rgb_array"):

        self.gym_factory = gym_factory
        self.history_size = history_size

        self._env = gym_factory.make(name, render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.height = resize_shape[0]
        self.width = resize_shape[1]

        self.state = self._env.reset()[0]
        self.reward = 0
        self.done = False
        self.info = None

        self.history = np.zeros([history_size+1,self.height,self.width], dtype=np.uint8)
        self.action_history = np.zeros([history_size+1], dtype=np.int32)
        self.number_lives = find_max_lifes(self._env)
        self.memory = None

        self.score = 0
        self.life = find_max_lifes(self._env)
        get_init_state(self.history, self.state, width=self.width, height=self.height)


    def step(self, action):
        state, reward, done, _, info = self._env.step(action)
        return state, reward, done, info

    def reset(self):
        self.history = np.zeros([self.history_size+1,self.height,self.width],
                                dtype=np.uint8)
        self.action_history = np.zeros([self.history_size+1], dtype=np.int32)
        return self._env.reset()[0]

    def render(self):
        self._env.render()

    def reset_memory(self, init):
        self.memory = init
        self.memory.requires_grad=False

    def get_available_actions(self):
        if self.gym_factory == gym:
            availability = np.ones(self.action_space.n)
        else:
            availability = self._env.get_available_actions()

        assert(not (len(np.unique(availability)) == 1 and not availability[0]))
        return availability
