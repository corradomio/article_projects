from collections import defaultdict

import networkx as nx

from utils import *


#
#   root
#       algo
#           dataset
#               deg
#                   graph
#                       tensor[10,deg,deg]

class CountGraphs:

    def __init__(self):
        self.count = defaultdict(lambda : 0)
        self.graphs: set[str] = set()

    # {
    #     'graph_id': gid,
    #     'n': n,
    #     'm': m,
    #     'adjacency_matrix': adjacency_matrix,
    #     'causal_discovery_algorithm': algorithm,
    #     'data_generation_method': (method, semtype),
    #     'dataset_index': i,
    #     'causal_adjacency_matrix': causal_dag,
    # }
    def add(self, info):
        gid = info["graph_id"]
        if gid not in self.graphs:
            n = info['n']
            self.graphs.add(gid)
            self.count[n] += 1

    def print(self):
        for d in self.count.keys():
            print(f"{d:2}: {self.count[d]}")


class CountMatches:
    def __init__(self):
        self.n_total = 0
        self.n_processed = defaultdict(lambda : 0)
        self.count = defaultdict(lambda : 0)
        self.disconnected = defaultdict(lambda : 0)
        self.not_a_dag = defaultdict(lambda : 0)
        self.zeros = defaultdict(lambda: 0)

    # {
    #     'graph_id': gid,
    #     'n': n,
    #     'm': m,
    #     'adjacency_matrix': adjacency_matrix,
    #     'causal_discovery_algorithm': algorithm,
    #     'data_generation_method': (method, semtype),
    #     'dataset_index': i,
    #     'causal_adjacency_matrix': causal_dag,
    # }
    def add(self, info):
        n = info['n']
        algorithm = info["causal_discovery_algorithm"]
        adjacency_matrix = info["adjacency_matrix"]
        causal_adjacency_matrix = info["causal_adjacency_matrix"]

        self.n_total += 1
        self.n_processed[(algorithm, n)] += 1

        if abs(adjacency_matrix - causal_adjacency_matrix).sum() == 0:
            self.count[(algorithm, n)] += 1
        elif causal_adjacency_matrix.sum() == 0:
            self.zeros[(algorithm, n)] += 1
            return

        G = nx.from_numpy_array(causal_adjacency_matrix, create_using=nx.DiGraph())
        if not nx.is_weakly_connected(G):
            self.disconnected[(algorithm, n)] += 1
        elif not nx.is_directed_acyclic_graph(G):
            self.not_a_dag[(algorithm, n)] += 1
        else:
            pass
        return

    def print(self):
        print(f"processed {self.n_total} graphs")
        for degree in self.n_processed:
            print(f"    {degree}: {self.n_processed[degree]}")
        print(f"found ground truth graphs")
        for degree in self.count:
            print(f"    {degree}: {self.count[degree]}")
        print(f"found disconnected graphs")
        for degree in self.disconnected:
            print(f"    {degree}: {self.disconnected[degree]}")
        print(f"found not dag graphs")
        for degree in self.not_a_dag:
            print(f"    {degree}: {self.not_a_dag[degree]}")
        print(f"found empty graphs")
        for degree in self.zeros:
            print(f"    {degree}: {self.zeros[degree]}")
        print("end")
# end


def main():
    tprint("Starting analysis ...", force=True)

    # foreach_datasets("./data")

    # cg = CountGraphs()
    # foreach_datasets("./data", callback=lambda path, info: cg.add(info))
    # cg.print()

    cm = CountMatches()
    foreach_dataset(
        "./data",
        callback=lambda path, info: cm.add(info),
        # max_degree=10,
        # skip_algos=['DirectLiNGAM', 'GES', 'GOLEM', 'ICALiNGAM', 'Notears']
    )
    cm.print()

    tprint("Done", force=True)
    pass


if __name__ == "__main__":
    # logging.config.fileConfig("logging_config.ini")
    # logging.getLogger("main").info("Logging initialized")
    main()
