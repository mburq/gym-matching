import gym
import numpy as np
from gym import spaces
import networkx as nx
import pandas as pd
from gym_matching.envs.matching import MatchingEnv


class TaxiMatchingEnv(MatchingEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        MatchingEnv.__init__(self, max_edge_weight = 1, time_steps = 200, observation_shape = (11, ))
        self.df = pd.read_csv("gym_matching/data/taxi/rides.csv")
        self.max_ride_length = 10
        self.departure_probability = 0.05

    def _new_vertex(self):
        """
        Returns a 1-d embeding based on the trip distance
        """
        i = np.random.randint(len(self.df))
        # origin = (float(self.df[['pX']].loc[id]), float(self.df[['pY']].loc[id]))
        # dest = (float(self.df[['dX']].loc[id]), float(self.df[['dY']].loc[id]))
        dist = float(self.df[['dist']].loc[i])
        bucket_dist = int(min(dist, self.max_ride_length) * self.max_ride_length // self.max_ride_length)
        return (i, bucket_dist)

    def _euclidian_dist(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _edge_weight(self, node1, node2):
        origin1 = [float(self.df[['pX']].loc[node1]), float(self.df[['pY']].loc[node1])]
        dest1 = [float(self.df[['dX']].loc[node1]), float(self.df[['dY']].loc[node1])]
        origin2 = [float(self.df[['pX']].loc[node2]), float(self.df[['pY']].loc[node2])]
        dest2 = [float(self.df[['dX']].loc[node2]), float(self.df[['dY']].loc[node2])]
        oa_ob = self._euclidian_dist(origin1, origin2)
        oa_db = self._euclidian_dist(origin1, dest2)
        ob_da = self._euclidian_dist(origin2, dest1)
        da_db = self._euclidian_dist(dest1, dest2)
        aa = self._euclidian_dist(origin1, dest1)
        bb = self._euclidian_dist(origin2, dest2)
        abab = oa_ob + ob_da + da_db
        abba = oa_ob + bb + da_db
        baba = oa_ob + oa_db + da_db
        baab = oa_ob + aa + da_db
        aabb = aa + bb  # no match
        match_value = aabb - min(abab, abba, baba, baab, aabb)
        assert match_value <= aabb / 2
        return match_value
