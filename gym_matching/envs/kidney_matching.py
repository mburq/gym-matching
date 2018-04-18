import gym
import numpy as np
from gym import spaces
import networkx as nx
import pandas as pd
from gym_matching.envs.matching import MatchingEnv


class KidneyMatchingEnv(MatchingEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        MatchingEnv.__init__(self, max_edge_weight = 1, time_steps = 200, observation_shape = (16, ))
        self.compatibility_matrix = pd.read_csv('gym_matching/data/kidney/compatibility_matrix.txt', sep='\t', header=None)
        self.features = pd.read_csv("gym_matching/data/kidney/pairs_pra.csv")
        self.departure_probability = 0.05

    def _new_vertex(self):
        """
        Returns a 2-d type based on patient and donor blood types, and a vertex name
        """
        blood_types_dict = {"O": 0, "A": 1, "B": 2, "AB": 3}
        i = np.random.randint(len(self.features))
        return (i,
                (blood_types_dict[self.features.ix[i, "pABO"]] + \
                 4 * blood_types_dict[self.features.ix[i, "dABO"]])
                )

    def _edge_weight(self, node1, node2):
        return self.compatibility_matrix.ix[node1, node2]
