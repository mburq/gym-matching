import gym
import numpy as np
from gym import spaces
import networkx as nx
from gym_matching.utils.max_weight_matching import gurobi_max_weight_matching


class TypeMatchingEnvShadow(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_types = 10
        self.action_space = None
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_types,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_types,))
        (m, a, d) = self._random_graph(self.num_types)
        self.matrix = m
        self.arrivals = a
        self.departures = d

    def _step(self, action):
        assert self.action_space.contains(action), "{} ({}) invalid".format(action, type(action))
        if np.all(action <= self.state):
            act = np.maximum(action, 0).astype(int)  # Shadow prices are in [0,1].
            (matches, reward) = gurobi_max_weight_matching(self.matrix, self.state, self.num_types, act)
            self.state -= matches
        else:
            reward = 0
        self._new_departures()
        self._new_arrivals()
        self.status = self.time_steps_remaining <= 0
        self.time_steps_remaining -= 1
        return self.state, reward, self.status, {}

    def _reset(self):
        self.state = np.zeros(self.num_types)
        self.reward = 0
        self.time_steps_remaining = 100
        return self.state

    def _random_graph(self, S):
        G = nx.erdos_renyi_graph(S, 0.5)
        for e in G.edges():
                    G[e[0]][e[1]]["weight"] = np.random.random()
        mat = nx.to_numpy_matrix(G)
        arrivals = np.random.random(S)
        departures = np.random.random(S)
        return (mat, arrivals, departures)

    def _new_arrivals(self):
        self.state += 1*(np.random.random(self.num_types) < self.arrivals)

    def _new_departures(self):
        for i in range(self.num_types):
            for j in range(int(self.state[i])):
                if np.random.random() < self.departures[i]:
                    self.state[i] -= 1
