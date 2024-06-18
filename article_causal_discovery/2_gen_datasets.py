import os

import h5py
import networkx as nx
import numpy as np
import random as rnd

import castle
import stdlib.jsonx as jsx
import logging.config
from stdlib.tprint import tprint
from castle.datasets.simulator import set_random_seed
from castlex.iidsim import IIDSimulation
from joblib import Parallel, delayed

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

log = None


def generate_datasets(container: h5py.File, ginfo: dict, N: int = 1000):
    """

    :param W: weighted adjacency matrix
    :param D: number of datasets to generate
    :param n: number of data points to generate
    :return:
    """
    # global log
    # log = logging.getLogger('main')

    n = ginfo['n']
    m = ginfo['m']
    wl_hash = ginfo['wl_hash']

    tprint(f"... processing {wl_hash} (order:{n}, size:{m})")

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


def collect_graphs(n_parts=0):
    all_graphs = []

    graphs = jsx.load("data/graphs-enum.json")
    for order in graphs["graphs"].keys():
        n = int(order)
        if n > 10:
            continue
        graphs_n = graphs["graphs"][order]
        for ginfo in graphs_n:
            all_graphs.append(ginfo)

    rnd.shuffle(all_graphs)

    if n_parts > 0:
        n_graphs = len(all_graphs)
        part_size = (n_graphs + n_parts - 1)//n_parts
        i = 0
        all_graphs_parts = []
        while i < n_graphs:
            all_graphs_parts.append(all_graphs[i: min(i+part_size, n_graphs)])
            i += part_size
        all_graphs = all_graphs_parts
    return all_graphs


def process_graphs(graphs_n, N, index=0):
    #
    # Create HDF5 container
    #
    # global log
    log = logging.getLogger(f"p{index:02}")

    tprint(f"[{index:2}] Start processing on {len(graphs_n)} graphs ...")

    if index < 0:
        assert len(graphs_n) > 100
        gdspath = f"data/graphs-datasets.hdf5"
    else:
        gdspath = f"data/graphs-datasets-{index:02}.hdf5"

    if os.path.exists(gdspath):
        os.remove(gdspath)
    container = h5py.File(gdspath, 'w')

    #
    # scan the graphs. Note: the key is a string
    #
    count = 0
    n_graphs = len(graphs_n)
    for ginfo in graphs_n:
        generate_datasets(container, ginfo, N)
        count += 1
        tprint(f"[{index:2}] ... {count:4}/{n_graphs}")
        # break

    container.close()
    tprint(f"[{index}] Done")
# end


def parallel_process_graphs(graphs, N, N_JOBS):

    if N_JOBS <= 1:
        for i in range(len(graphs)):
            process_graphs(graphs[i], N, i)
    else:
        Parallel(n_jobs=N_JOBS)(
            delayed(process_graphs)(graphs[i], N, i) for i in range(N_JOBS)
        )


def serial_process_graphs(graphs, N):
    process_graphs(graphs, N, -1)


def main():
    global log
    log = logging.getLogger('main')
    tprint(f"nx: {nx.__version__}")
    tprint(f"castle: {castle.__version__}")

    # n of records in the dataset
    N = 10000
    N_JOBS = 8

    set_random_seed(42)

    #
    # Load graphs
    #
    # graphs = collect_graphs(n_parts=N_JOBS)
    # parallel_process_graphs(graphs, N, N_JOBS)

    graphs = collect_graphs()
    serial_process_graphs(graphs, N)
# end


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
