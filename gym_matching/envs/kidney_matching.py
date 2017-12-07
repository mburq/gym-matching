import gym
import numpy as np
from gym import spaces
from gym_matching.utils.max_weight_matching import gurobi_max_weight_matching
import networkx as nx
import pandas as pd


class KidneyMatchingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_types = 16
        # The state space is the compatibility graph. The algorithm only observes how many of each type is present.
        self.observation_space = spaces.Box(low=0, high=20, shape=(self.num_types,))
        # The agent's actions are only related to the future value of vertices (not the matching itself)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_types,))
        # TODO: make the low, high depend on max weight in type graph?
        self.compatibility_matrix = pd.read_csv('gym_matching/data/kidney/compatibility_matrix.txt', sep='\t', header=None)
        self.features = pd.read_csv("gym_matching/data/kidney/pairs_pra.csv")
        self.time_steps_max = 500
        self.departure_probability = 0.05

    def _step(self, action):
        assert self.action_space.contains(action), "{} ({}) invalid".format(action, type(action))
        act = [action[self._find_type(i)] for (i, t) in self.state.nodes()]
        matrix = nx.to_numpy_matrix(self.state)
        (matches, reward) = gurobi_max_weight_matching(matrix, np.ones(len(matrix)), len(matrix), act)
        to_remove = [v for (e, v) in enumerate(self.state.nodes()) if matches[e] == 1]
        self._remove_vertices(to_remove)
        self._new_departures()
        self._new_arrivals()
        self.status = self.time_steps >= self.time_steps_max
        self.time_steps += 1
        return self.types, reward, self.status, {}

    def _reset(self):
        self.types = np.zeros(self.num_types)
        self.state = nx.Graph()
        self.reward = 0
        self.time_steps = 0
        return self.types

    def _new_arrivals(self):
        i = np.random.randint(len(self.features))
        new_vertex = (i, self.time_steps)
        self.state.add_node(new_vertex)
        for v in self.state.nodes():
            match_weight = self.compatibility_matrix.ix[i, v[0]]
            if match_weight > 0 and v != new_vertex:
                self.state.add_edge(new_vertex, v, weight=match_weight)
        self.types[self._find_type(i)] += 1

    def _find_type(self, i):
        """
        Blood-type dependent vertex types
        """
        blood_types_dict = {"O": 0, "A": 1, "B": 2, "AB": 3}
        return blood_types_dict[self.features.ix[i, "pABO"]] + \
            4 * blood_types_dict[self.features.ix[i, "dABO"]]  # + \
            # 16 * (self.features.ix[i, "PRA"] >= 90)

    def _remove_vertices(self, to_remove):
        for v in to_remove:
            self.types[self._find_type(v[0])] -= 1
        self.state.remove_nodes_from(to_remove)

    def _new_departures(self):
        to_remove = []
        for v in self.state.nodes():
            if np.random.rand() < self.departure_probability:
                to_remove.append(v)
        self._remove_vertices(to_remove)
