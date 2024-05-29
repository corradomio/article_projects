import os
import random as rnd
from typing import Union

import castle
import h5py
import networkx as nx
import torch
import numpy as np
from joblib import Parallel, delayed

import numpyx as npx
import stdlib
import stdlib.jsonx as jsonx
import stdlib.loggingx as logging

log: logging.Logger = logging.getLogger("main")

GRAPH_ORDERS = ["2", "3", "4", "5", "10", "15", "20", "25"]

GRAPH_ALGORITHMS: dict[str, dict] = jsonx.load("data/graphs-algorithms.json")


def create_algo(algo_name: str, order: int = 0) -> castle.common.base.BaseLearner:
    """
    Create the specified algorithm with parameters specific for graphs with
    the selected order

    :param algo_name: algorithm name
    :param order: graph order (number of vertices)
    :return: an instance of the algorithm
    """
    algo_info = GRAPH_ALGORITHMS[algo_name]
    klass = algo_info["class"]
    if isinstance(klass, str):
        klass = stdlib.import_from(klass)
    sorder = str(order)
    if sorder in algo_info:
        kwparams = algo_info[sorder]
    else:
        kwparams = {} | GRAPH_ALGORITHMS[algo_name]
        # remove all extra parameters:
        for key in ["class"] + GRAPH_ORDERS:
            if key in kwparams:
                del kwparams[key]
    # end
    return klass(**kwparams)


DATA_PREPARATION = {
    "ANMNonlinear": ["float64", "mean0"],
}


def prepare_data(algo_name, X: np.ndarray) -> np.ndarray:
    # Some algorithms require the data in some special format.
    # For example:
    #   - it is better to have the data in 'float64' instead than 'float32'
    #   - the data must be 'mean=0, standard_deviation=1'
    #   - etc
    if algo_name not in DATA_PREPARATION:
        return X

    for method in DATA_PREPARATION[algo_name]:
        if method == "mean0":
            # mean=0/standard_deviation=1
            scaler = npx.scalers.StandardScaler()
            X = scaler.fit_transform(X)
        elif method == "float64":
            # dataset in format float64/double
            X = X.astype(np.float64)
    return X


# All dataset have 10000 records and can be considered as the concatenation
# of 10 different datasets with 1000 records each one

DS_LEN = 10000
DS_PART_LEN = 1000


def list_graph_names(n_parts: int = 1) -> Union[list[str], list[list[str]]]:
    """
    Collect all graphs names available in 'c'
    Note: the graphs are located in 'c' in increasing order or vertices' counts
        When used with 'Parallel', it is better to shuffle the names in such way
        more or less all Python instances will process the assigned graphs in
        similar time.
    """

    with h5py.File("data/graphs-datasets.hdf5", "r") as c:
        # collect the graph names
        graph_names = []
        for order in GRAPH_ORDERS:
            graphs_list = c[order]

            for graph_id in graphs_list.keys():
                graph_names.append(graphs_list[graph_id].name)

    # sequential processing
    if n_parts <= 1:
        return graph_names

    # parallel processing: split the list in 'parts' sub-lists of
    # similar size except the last one
    rnd.shuffle(graph_names)
    part_len = (len(graph_names) + n_parts - 1) // n_parts

    graph_names_lists = []
    for i in range(n_parts - 1):
        s = i*part_len
        graph_names_lists.append(graph_names[s:s+part_len])
    s = (n_parts - 1) * part_len
    graph_names_lists.append(graph_names[s:])

    return graph_names_lists
# end


def process_graphs(parallel=False):
    # collect the list of graphs
    # can be executed in parallel

    if parallel:
        # n_jobs = 8
        # n_parts = n_jobs * 2

        n_jobs = 10
        n_parts = n_jobs

        graph_names_lists = list_graph_names(n_parts=n_parts)
        Parallel(n_jobs=n_jobs)(delayed(parallel_process_graphs)(i + 1, graph_names_lists[i]) for i in range(n_parts))
    else:
        sequential_process_graphs(list_graph_names())
    return
# end


def sequential_process_graphs(graph_names: list[str]):
    log.info(f"Process sequential started on {len(graph_names)} graphs")
    c = h5py.File("data/graphs-datasets.hdf5", "r")

    if os.path.exists("data/graphs-predictions.hdf5"):
        os.remove("data/graphs-predictions.hdf5")
    p = h5py.File("data/graphs-predictions.hdf5", "w")

    # list of order/size graphs processed
    processed = set()

    for graph_name in graph_names:
        graph_info = c[graph_name]
        n = graph_info.attrs['n']
        m = graph_info.attrs['m']

        # FILTER
        # process a single graph with the specified (order, size)
        pair = (n, m)
        if pair in processed:
            continue
        else:
            processed.add(pair)
        # FILTER END

        process_graph(graph_info, p)
        # break       # FILTER
    return
# end


def parallel_process_graphs(process_id, graph_names: list[str]):
    # Initialize the logging system in THIS process
    global log
    logging.config.fileConfig("logging_config.ini")
    log = logging.getLogger(f"main.p{process_id:02}")
    log.info(f"Process {process_id:02} started on {len(graph_names)} graphs")

    # load the datasets
    c = h5py.File("data/graphs-datasets.hdf5", "r")

    # initialize the container for the results
    graph_predictions_name = f"data/graphs-predictions-{process_id:02}.hdf5"
    if os.path.exists(graph_predictions_name):
        os.remove(graph_predictions_name)
    p = h5py.File(graph_predictions_name, "w")

    # scan the graphs
    for graph_name in graph_names:
        graph_info = c[graph_name]
        process_graph(graph_info, p)
        # break
    return
# end


def process_graph(graph_info: h5py.Group, p: h5py.Group = None):
    name = graph_info.name
    n = graph_info.attrs["n"]
    m = graph_info.attrs["m"]
    wl_hash = graph_info.attrs["wl_hash"]

    log.info(f"... {name}: {n} x {m}")

    A = graph_info.attrs["adjacency_matrix"]

    pred_info = p.create_group(name)
    pred_info.attrs["n"] = n
    pred_info.attrs["m"] = m
    pred_info.attrs["adjacency_matrix"] = A
    pred_info.attrs["wl_hash"] = wl_hash

    # scan algorithms
    for algo_name in GRAPH_ALGORITHMS:
        # skip the algorithms starting with #
        if algo_name.startswith("#"): continue

        # scan sem_type
        for sem_type in graph_info.keys():
            log.info(f"... ... {algo_name}/{sem_type}")

            datasets: h5py.Dataset = graph_info[sem_type]
            method = datasets.attrs["method"]
            nds = DS_LEN // DS_PART_LEN

            CM = np.zeros((nds, n, n), dtype=np.int8)
            # scan datasets
            for i in range(nds):
                s = i*DS_PART_LEN
                X = datasets[s:s+DS_PART_LEN, :]

                # create the algorithm for graphs with the specified order
                algo = create_algo(algo_name, order=n)
                # some algorithms need data in some special format
                X = prepare_data(algo_name, X)

                algo.learn(X)
                C: np.ndarray = algo.causal_matrix.astype(np.int8)

                CM[i, :, :] = C

                break   # FILTER
            # end
            dset = p.create_dataset(f"{name}/{algo_name}/{sem_type}", (nds, n, n), dtype=CM.dtype, data=CM)
            dset.attrs["method"] = method
            dset.attrs["sem_type"] = sem_type
            dset.attrs["algorithm"] = algo_name
            break       # FILTER
        break           # FILTER
    # end
# end


def main():
    global log
    log = logging.getLogger("main")
    log.info(f"nx: {nx.__version__}")
    log.info(f"castle: {castle.__version__}")
    log.info(f"torch: {torch.__version__}")

    # Castle:
    # You can use `os.environ['CASTLE_BACKEND'] = backend` to set the backend(`pytorch` or `mindspore`).
    # Note: 'mindspore' is available ONLY for Python 3.8/3.9 AND it is Chinese package
    os.environ["CASTLE_BACKEND"] = "pytorch"

    process_graphs(parallel=False)

    log.info("done")
# end


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
