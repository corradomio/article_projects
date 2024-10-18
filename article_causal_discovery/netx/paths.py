from .graph import Graph

class Node:
    def __init__(self, n: int, prev: list[int]=[]):
        self.path = prev + [n]
        self.node = n



def shortest_path(G: Graph, u: int, v: int) -> list[int]:
    toprocess = [Node(u)]
    processed = set()
    while len(toprocess) > 0:
        u_node = toprocess.pop()
        u = u_node.node
        processed.add(u)
        u_path = u_node.path
        tlist = G.succ[u]
        for t in tlist:
            t_node = Node(t, u_path)
            if t == v:
                return t_node.path
            if t in processed:
                continue
            toprocess.append(t_node)
        # end
    # end
    return []
# end
