import h5py
import numpy as np
from path import Path as path
from typing import Union, Optional


#
#   [data_root: directory]
#       [algorithm: directory]
#           graphs-predictions-<index>.hdf5: file
#               [degree: h5py.Group]
#                   [graph_id: h5py.Group]
#                       attributes:
#                           adjacency_matrix: shape=(<degree>, <degree>),  type=int8
#                           n: <degree>
#                           m: <number of edges>
#                       [algorithm: h5py.Group]
#                           [data_distribution: h5py.Dataset]
#                               - shape=(10, <degree>, <degree>)
#                               - type=int8
#


def scan_graphs(data_root: str, degree: Union[None, int, tuple[int, int]] = None):
    if degree is None:
        dmin, dmax = 0, 100
    elif isinstance(degree, int):
        dmin, dmax = 0, degree
    else:
        dmin, dmax = degree

    for dir_algo in path(data_root).dirs():
        for hdf in dir_algo.files("*.hdf5"):
            print(hdf)
            with h5py.File(hdf, mode='r') as f:
                # scan the degrees (is a str):
                for sdeg in f:
                    deg = int(sdeg)
                    if deg < dmin or deg > dmax:
                        # skip out of range
                        continue

                    # scan the graphs
                    grp_deg: h5py.Group = f[sdeg]
                    for gid in grp_deg:
                        grp_graph = grp_deg[gid]

                        n_nodes = grp_graph.attrs['n']
                        n_edges = grp_graph.attrs['m']
                        adjacency_matrix: np.ndarray = grp_graph.attrs['adjacency_matrix']
                        ground_truth = adjacency_matrix

                        # scan the algorithms
                        for graph_algo in grp_graph:
                            grp_algo = grp_graph[graph_algo]

                            # scan the data distributions
                            for data_distrib in grp_algo:
                                dataset = grp_algo[data_distrib]

                                # numpy tensor: (10, n, n)
                                for i in range(10):
                                    causal_graph: np.ndarray = dataset[i]

                                    print(dir_algo.stem, hdf.stem, deg, gid, data_distrib, i)
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        pass
    pass
# end


def main():
    scan_graphs("../article_causal_discovery_data", degree=5)
    pass


if __name__ == "__main__":
    main()
