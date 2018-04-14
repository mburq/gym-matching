import gym
import numpy as np
from gym import spaces
import networkx as nx
import pandas as pd


class MatchingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, vertex_generator, observation_shape = (4, 4)):
        self.observation_shape = observation_shape
        # The state space is the compatibility graph. The algorithm only observes how many of each type is present.
        self.observation_space = spaces.Box(low=0, high=20, shape=observation_shape)
        # The agent's actions are only related to the future value of vertices (not the matching itself)
        self.action_space = spaces.Box(low=-1, high=1, shape=observation_shape)
        # TODO: make the low, high depend on max weight in type graph?
        self.time_steps_max = 500
        self.departure_probability = 0.05
        self.vertex_generator = vertex_generator
        # An object that generates vertices and edges. Problem specific.

    def _step(self, action):
        assert self.action_space.contains(action), "{} ({}) invalid".format(action, type(action))
        act = [action[self._find_type(i)] for (i, t) in self.state.nodes()]
        self._update_graph_weights(act)
        matches, reward = nx.max_weight_matching(self.graph)
        to_remove = [v for (e, v) in enumerate(self.state.nodes()) if matches[e] == 1]
        self._remove_vertices(to_remove)
        self._new_departures()
        self._new_arrivals()
        self.status = self.time_steps >= self.time_steps_max
        self.time_steps += 1
        return self.types, reward, self.status, {}

    def _reset(self):
        self.observation = np.zeros(self.observation_shape)
        self.graph = nx.Graph()
        self.reward = 0
        self.time_steps = 0
        return self.types

    def _new_arrivals(self):
        new_vertex, embed = self.vertex_generator.new_vertex()
        self.state.add_node(new_vertex, arr_time=self.time_steps, embeding=embed)
        for v in self.state.nodes():
            match_weight = self.vertex_generator.edge_weight(new_vertex, v)
            if match_weight > 0 and v != new_vertex:
                self.state.add_edge(new_vertex, v, edge_weight=match_weight)
        self.observation[embed] += 1

    def _update_graph_weights(self, action):
        for e in self.state.edges():
            self.state[e]['weight'] = self.state[e]['edge_weight'] - \
                action[e[0]['embeding']] - action[e[1]['embeding']]

    def _remove_vertices(self, to_remove):
        for v in to_remove:
            self.observation[v['embeding']] -= 1
        self.state.remove_nodes_from(to_remove)

    def _new_departures(self):
        to_remove = []
        for v in self.state.nodes():
            if np.random.rand() < self.departure_probability:
                to_remove.append(v)
        self._remove_vertices(to_remove)
