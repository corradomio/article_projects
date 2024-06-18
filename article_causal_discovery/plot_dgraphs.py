import logging.config

import matplotlib.pyplot as plt

# import netx as nx
import networkx as nx
from utils import *


#
#   root
#       algo
#           dataset
#               deg
#                   graph
#                       tensor[10,deg,deg]

class PlotGraph:

    def __init__(self, max_samples=20):
        self.max_samples = max_samples
        pass

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
        n = info["n"]
        m = info["m"]

        droot = path(f"discovered/{n}")
        groot = path(f"discovered/{n}").joinpath(gid)
        if (self.max_samples != 0
            and droot.exists()
            and not groot.exists()
            and len(droot.dirs()) >= self.max_samples):
            return

        dataset_index = info["dataset_index"]

        adjacency_matrix = info["adjacency_matrix"]
        causal_adjacency_matrix = info["causal_adjacency_matrix"]

        causal_discovery_algorithm = info["causal_discovery_algorithm"]
        _, data_generation_method = info["data_generation_method"]

        ground_truth = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())
        causal_graph = nx.from_numpy_array(causal_adjacency_matrix, create_using=nx.DiGraph())

        # create the root directory:
        # root = path(f"discovered/{n}/{gid}")
        gtname = groot.joinpath("ground_truth.png")
        if not gtname.exists():
            groot.makedirs_p()

            plt.clf()
            plt.gca()
            nx.draw(ground_truth, pos=nx.circular_layout(ground_truth), with_labels=True)
            plt.title(f"Ground Truth: |V|={n}, |E|={m}")
            plt.savefig(gtname, dpi=300)

        # create the directory for the predictions
        # pred = path(f"discovered/{n}/{gid}/{causal_discovery_algorithm}")
        pred = groot.joinpath(causal_discovery_algorithm)
        pred.makedirs_p()

        # fname = pred.joinpath(f"{causal_discovery_algorithm}-{data_generation_method}-{dataset_index}.png")
        fname = pred.joinpath(f"{data_generation_method}-{dataset_index}.png")
        if fname.exists():
            return

        plt.clf()
        plt.gca()
        nx.draw(causal_graph, pos=nx.circular_layout(causal_graph), with_labels=True)
        plt.title(f"{causal_discovery_algorithm}/{data_generation_method}[{dataset_index}]: |V|={n}, |E|={causal_graph.size()}")

        plt.savefig(fname, dpi=300)
    # end
# end


def main():
    tprint("Starting plot ...", force=True)

    pg = PlotGraph(max_samples=10)
    # foreach_dataset("./data", callback=lambda path, info: pg.add(info), max_degree=5)
    parallel_foreach_dataset("./data", callback=lambda path, info: pg.add(info), max_degree=10, n_jobs=6)

    tprint("Done", force=True)
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
