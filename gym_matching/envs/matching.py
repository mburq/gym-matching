import gym
import numpy as np
from gym import spaces
import networkx as nx
import pandas as pd
from gym.utils import seeding


class MatchingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 time_steps = 50,
                 max_edge_weight = 10,
                 departure_probability = 0.,
                 observation_shape = (2,)
                 ):
        self.observation_shape = observation_shape
        self.max_edge_weight = max_edge_weight
        # The state space is the compatibility graph. The algorithm only observes how many of each type is present.
        self.observation_space = spaces.Box(low=0, high=20, shape=self.observation_shape)
        # The agent's actions are only related to the future value of vertices (not the matching itself)
        self.action_space = spaces.Box(low=-self.max_edge_weight / 2, high=self.max_edge_weight / 2, shape=self.observation_shape)
        # TODO: make the low, high depend on max weight in type graph?
        self.time_steps_max = time_steps
        self.departure_probability = departure_probability
        # An object that generates vertices and edges. Problem specific.

    def step(self, action):
        """
        Args:
         - `action`: A vector with dimension `observation_shape` that maps an embeding
         to a scalar value that represents the `shadow price` of a given vertex.
        """
        assert self.action_space.contains(action), "{} ({}) invalid".format(action, type(action))
        self._update_graph_weights(action + self.max_edge_weight / 2)
        matches = nx.max_weight_matching(self.graph)
        reward = sum([self.graph[i][j]['edge_weight'] for (i,j) in matches])
        to_remove = [i for (i,j) in matches] + [j for (i,j) in matches]
        # TODO: check that the order is always the same in pairs in matches.
        # [v for (e, v) in enumerate(self.graph.nodes()) if matches[e] == 1]
        self._remove_vertices(to_remove)
        self._new_departures()
        self._new_arrivals()
        self.status = self.time_steps >= self.time_steps_max
        self.time_steps += 1
        return self.observation, reward, self.status, {}

    def reset(self):
        self.observation = np.zeros(self.observation_shape)
        self.graph = nx.Graph()
        self.reward = 0
        self.time_steps = 0
        return self.observation

    def _new_vertex(self):
        """
        Returns a vertex that can take two possible types (H, E)
        """
        if np.random.random() < 0.8:
            t = 0
        else:
            t = 1
        return (t, t)

    def _edge_weight(self, n1, n2):
        if self.graph.nodes[n1]['embeding'] == 1 and self.graph.nodes[n2]['embeding'] == 1:
            return 10
        else:
            return 0.1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _new_arrivals(self):
        new_vertex_features, embed = self._new_vertex()
        new_vertex = self.time_steps
        self.graph.add_node(new_vertex, features = new_vertex_features,  arr_time=self.time_steps, embeding=embed)
        for old_vertex in self.graph.nodes:
            match_weight = self._edge_weight(old_vertex, new_vertex)
            if match_weight > 0 and old_vertex != new_vertex:
                self.graph.add_edge(new_vertex, old_vertex, edge_weight=match_weight)
        self.observation[embed] += 1

    def _update_graph_weights(self, action):
        for e in self.graph.edges():
            self.graph.edges[e]['weight'] = self.graph.edges[e]['edge_weight'] - \
                action[self.graph.nodes[e[0]]['embeding']] - \
                action[self.graph.nodes[e[1]]['embeding']]

    def _remove_vertices(self, to_remove):
        for v in to_remove:
            self.observation[self.graph.nodes[v]['embeding']] -= 1
        self.graph.remove_nodes_from(to_remove)

    def _new_departures(self):
        to_remove = []
        for v in self.graph.nodes():
            if np.random.rand() < self.departure_probability:
                to_remove.append(v)
        self._remove_vertices(to_remove)
