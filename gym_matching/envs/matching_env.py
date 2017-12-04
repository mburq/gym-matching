import gym
import numpy as np
from gym import spaces
import networkx as nx
import gurobipy as gb


class TypeMatchingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_types = 10
        self.action_space = None
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_types,))
        # self.action_space = spaces.Box(low=0, high=100, shape=(self.num_types,))
        # Discrete action space so we can run DQN. An action is a match and whether to match again.
        self.action_space = spaces.Discrete(2 * self.num_types**2)
        (m, a, d) = self._random_graph(self.num_types)
        self.matrix = m
        self.arrivals = a
        self.departures = d

    # def _step(self, action):
    #     assert self.action_space.contains(action), "{} ({}) invalid".format(action, type(action))
    #     assert np.all(self.action >= self.state)
    #     (matches, reward) = self._gurobi_max_weight_matching(self.matrix, action, self.num_types)
    #     self.state -= matches
    #     self._new_departures()
    #     self._new_arrivals()
    #     self.status = self.time_steps_remaining <= 0
    #     self.time_steps_remaining -= 1
    #     return self.state, reward, self.status, {}
    def _step(self, action):
        reward = 0
        m1 = (action // 2) % self.num_types
        m2 = (action // 2) // self.num_types
        keep_matching = (action % 2 == 1)
        if self.state[m1] >= 1 and self.state[m2] >= 1:
            self.state[m1] -= 1
            self.state[m2] -= 1
            reward += self.matrix[m1, m2]
        if keep_matching == 0:
            self._new_departures()
            self._new_arrivals()
            self.time_steps_remaining -= 1
        self.status = self.time_steps_remaining <= 0
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

    def _gurobi_max_weight_matching(self, matrix, state, S):
        """
        Implements a max weight matching MIP, returns a max-weight matching of the current graph.
        """
        vertex_discount = np.zeros(S)  # TODO: enable discounting
        m = gb.Model("max_weight_matching")
        n = len(matrix)
        assert n == S
        x = {}  # dictionnary of all the variables in the MIP
        for i in range(S):
            x[(i)] = m.addVar(obj=-vertex_discount[i], vtype=gb.GRB.INTEGER, name="y{}".format(i))
            for j in range(i+1, S):
                x[(i, j)] = m.addVar(obj=matrix[i, j], vtype=gb.GRB.BINARY, name="({},{})".format(i, j))
        m.update()
        constraints = []
        for i in range(S):
            c = sum(x[(i, j)] for j in range(i+1, S)) + sum(x[(j, i)] for j in range(i))
            constraints.append(m.addConstr(c == x[(i)], "matching {}".format(i)))
            constraints.append(m.addConstr(x[(i)] <= state[i], "matching {}".format(i)))
        m.update()
        m.modelSense = gb.GRB.MAXIMIZE
        m.update()
        m.params.OutputFlag = 0
        m.optimize()
        return (np.array([x[(i)].x for i in range(S)]).astype(int), m.objVal)
