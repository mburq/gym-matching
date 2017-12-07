import gym
import numpy as np
from gym import spaces
from gym_matching.utils.max_weight_matching import gurobi_max_weight_matching
from gym_matching.utils.type_graphs import random_graph, bomb_graph


class TypeMatchingEnvShadow(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_types = 10
        self.observation_space = spaces.Box(low=0, high=20, shape=(self.num_types,))
        self.action_space = spaces.Box(low=-11, high=11, shape=(self.num_types,))
        # TODO: make the low, high depend on max weight in type graph?
        # (m, a, d) = random_graph(self.num_types)
        (m, a, d) = bomb_graph(self.num_types)
        self.matrix = m
        self.arrivals = a
        self.departures = d

    def _step(self, action):
        assert self.action_space.contains(action), "{} ({}) invalid".format(action, type(action))
        act = np.add(action, 0)  # Shadow prices are in [0,1].
        (matches, reward) = gurobi_max_weight_matching(self.matrix, self.state, self.num_types, act)
        self.state -= matches

        self._new_departures()
        self._new_arrivals()
        self.status = self.time_steps_remaining <= 0
        self.time_steps_remaining -= 1
        return self.state, reward, self.status, {}

    def _reset(self):
        self.state = np.zeros(self.num_types)
        self.reward = 0
        self.time_steps_remaining = 500
        return self.state

    def _new_arrivals(self):
        self.state += 1*(np.random.random(self.num_types) < self.arrivals)

    def _new_departures(self):
        for i in range(self.num_types):
            for j in range(int(self.state[i])):
                if np.random.random() < self.departures[i]:
                    self.state[i] -= 1
