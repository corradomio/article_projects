# WARNING:
#
#   based on netx.Graph
#   NOT   on networkx.Graph
#

from collections import deque
from typing import Iterator

from .graph import Graph

#
# G.cache: dict[str, dict[...]]
#


def sources(G: Graph) -> set[int]:
    if "nodes" not in G.cache["sources"]:
        slist = []
        for n in G.nodes_:
            if G.in_degree(n) == 0:
                slist.append(n)

        G.cache["sources"]["nodes"] = set(slist)
    return G.cache["sources"]["nodes"]


def destinations(G: Graph) -> set[int]:
    if "nodes" not in G.cache["destinations"]:
        dlist = []
        for n in G.nodes_:
            if G.out_degree(n) == 0:
                dlist.append(n)
        G.cache["destinations"]["nodes"] = set(dlist)
    return G.cache["destinations"]["nodes"]


def parents(G: Graph, n: int) -> set[int]:
    return set(G.pred[n])


def children(G: Graph, n: int) -> set[int]:
    return set(G.succ[n])


def ancestors(G: Graph, v: int, include=False) -> set[int]:
    cname = "ancestors_incl" if include else "ancestors"
    if v not in G.cache[cname]:
        waiting = deque()
        ancestors: set[int] = set()
        if include: ancestors.add(v)
        waiting.extend(parents(G, v))
        while waiting:
            u = waiting.popleft()
            if u not in ancestors:
                ancestors.add(u)
                waiting.extend(G.pred[u])
        G.cache[cname][v] = ancestors
    # end
    return G.cache[cname][v]


def descendants(G: Graph, u, include=False) -> set[int]:
    cname = "descendants_incl" if include else "descendants"
    if u not in G.cache[cname]:
        waiting = deque()
        descendants: set[int] = set()
        if include: descendants.add(u)
        waiting.extend(children(G, u))
        while waiting:
            v = waiting.popleft()
            if v not in descendants:
                descendants.add(v)
                waiting.extend(G.succ[v])
        G.cache[cname][u] = descendants
    # end
    return G.cache[cname][u]


# ---------------------------------------------------------------------------

def _find_path(G: Graph, u_path: list[int], s: int, v: int, u_processed: set[int]) \
        -> Iterator[list[int]]:
    if s == v:
        yield u_path + [v]

    if s in u_processed:
        return

    for t in G.succ[s]:
        yield from _find_path(G, u_path + [s], t, v, u_processed | {s})


def find_paths(G: Graph, u: int, v: int) -> Iterator[list[int]]:
    u_processed = {u}
    for t in G.succ[u]:
        yield from _find_path(G, [u], t, v, u_processed)


def find_all_paths(G: Graph, u: int, v: int) -> list[list[int]]:
    uv = u, v
    if uv not in G.cache["find_paths"]:
        uv_paths = list(find_paths(G, u, v))
        G.cache["find_paths"][uv] = uv_paths
    return G.cache["find_paths"][uv]
