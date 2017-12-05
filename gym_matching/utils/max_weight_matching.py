import gurobipy as gb
import numpy as np


def gurobi_max_weight_matching(matrix, state, S, vertex_discount=None):
    """
    Implements a max weight matching MIP, returns a max-weight matching of the current graph.
    """
    if vertex_discount is None:
        vertex_discount = np.zeros(S)
    assert len(vertex_discount) == S

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
