import numpy as np

from . import from_numpy_array
from .graph import Graph


def from_numpy_matrix(A) -> Graph:
    return from_numpy_array(A)


def _is_directed(A: np.ndarray) -> bool:
    n, m = A.shape
    for i in range(n-1):
        for j in range(i+1, m):
            if A[i,j] != A[j,i]:
                return True
    return False


def _has_loops(A: np.ndarray) -> bool:
    n, m = A.shape
    for i in range(n):
        if A[i, i] != 0:
            return True
    return False


def from_numpy_array(A: np.ndarray):
    direct =_is_directed(A)
    loops=_has_loops(A)
    multi=False
    acyclic=False # to check

    n, m = A.shape
    nodes = list(range(n))
    edges: list[tuple[int, int]] = []
    for u in nodes:
        for v in nodes:
            if A[u, v]:
                edges.append((u, v))

    G = Graph(direct=direct, loops=loops, multi=multi, acyclic=acyclic)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G
# end
