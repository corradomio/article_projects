import os

import castle
import h5py
import networkx as nx
import numpy as np
import random as rnd

import stdlib.jsonx as jsx
import stdlib.loggingx as logging
import castlex
from castlex.iidsim import IIDSimulation


# IIDSimulation
#     W: np.ndarray
#         Weighted adjacency matrix for the target causal graph.
#     n: int
#         Number of samples for standard trainning dataset.
#     linear
#         sem_type: gauss, exp, gumbel, uniform, logistic
#     nonlinear
#         sem_type: mlp, mim, gp, gp-add, quadratic
#     noise_scale
#         Scale parameter of noise distribution in linear SEM

SEM_TYPES = {
    "linear": ["gauss", "exp", "gumbel", "uniform", "logistic"],
    "nonlinear": ["mlp", "mim", "quadratic"]  # "gp", "gp-add",
}


def generate_datasets(container: h5py.File, ginfo: dict, N: int = 1000):
    """

    :param W: weighted adjacency matrix
    :param D: number of datasets to generate
    :param n: number of data points to generate
    :return:
    """
    log = logging.getLogger('main')

    n = ginfo['n']
    m = ginfo['m']
    wl_hash = ginfo['wl_hash']

    log.info(f"processing {wl_hash} (order:{n}, size:{m})")

    W: np.ndarray = np.array(ginfo["adjacency_matrix"]).astype(np.int8)

    # ds = container.create_dataset(f"{n}/{wl_hash}/adjacency_matrix", (n, n), dtype=W.dtype, data=W)
    grp = container.create_group(f"{n}/{wl_hash}")
    grp.attrs['n'] = n
    grp.attrs['m'] = m
    grp.attrs['wl_hash'] = wl_hash
    grp.attrs['adjacency_matrix'] = W

    for method in SEM_TYPES:
        for sem_type in SEM_TYPES[method]:
            # log.info(f"... {method}/{sem_type}")

            iidsim = IIDSimulation(method=method, sem_type=sem_type)
            iidsim.fit(W)
            ds = iidsim.generate(N).astype(np.float32)

            dset = container.create_dataset(
                f"{n}/{wl_hash}/{sem_type}", (N, n), dtype=ds.dtype, data=ds)
            dset.attrs['method'] = method
            dset.attrs['sem_type'] = sem_type
            # break
        # break
    # end
    return
# end


def main():
    log = logging.getLogger('main')
    log.info(f"nx: {nx.__version__}")
    log.info(f"castle: {castle.__version__}")

    # n of records in the dataset
    N = 10000

    rnd.seed(42)
    castlex.set_random_seed(42)

    #
    # Load graphs
    #
    graphs = jsx.load("data/graphs-enum.json")

    #
    # Create HDF5 container
    #

    if os.path.exists('data/graphs-datasets.hdf5'):
        os.remove('data/graphs-datasets.hdf5')
    container = h5py.File('data/graphs-datasets.hdf5', 'w')

    #
    # scan the graphs. Note: the key is a string
    #
    for order in graphs["graphs"].keys():
        graphs_n = graphs["graphs"][order]

        for ginfo in graphs_n:
            generate_datasets(container, ginfo, N)
            # break

    container.close()
# end


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
