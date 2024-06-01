import os
import random as rnd
from typing import Union, Optional

import castle
import h5py
import networkx as nx
import torch
import numpy as np
from path import Path as path
from joblib import Parallel, delayed

import numpyx as npx
import stdlib
import stdlib.jsonx as jsonx
import stdlib.loggingx as logging
from stdlib import is_instance

log: logging.Logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

# Configuration file for the algorithms to use
GRAPH_ALGO_CONFIG = "data/graphs-algorithms.json"

# Name of the file containing the datasets
GRAPH_DATASETS = "data/graphs-datasets.hdf5"

# Graph orders. In theory this list is not necessary,
# but it is used for historical reasons
GRAPH_ORDERS = ["2", "3", "4", "5", "10", "15", "20", "25"]

# Dictionary containing the algorithms' configurations
GRAPH_ALGORITHMS: dict[str, dict] = jsonx.load(GRAPH_ALGO_CONFIG)

# Extra configurations for specific algorithms
DATA_PREPARATION = {
    "ANMNonlinear": ["float64", "mean0"],
}

# All datasets have 10000 records. The can be considered as the
# concatenation of 10 datasets of 1000 records each one.
# But it is possible to select different configurations
DS_LEN = 10000
DS_PART_LEN = 1000

# N of jobs for parallelism. If N_JOBS is less or equals to 1,
# it is used a sequential analysis
N_JOBS = 10
N_PARTS = N_JOBS

# It is possible to exclude graphs with specific degrees or datasets
# generated in some specific way. For performance/exception reasons
# we skip graphs with 25 nodes and dataset generated using 'logistic'
# method.
EXCLUDE_GRAPH_DEGREES = [25]
EXCLUDE_SEM_TYPES = ["logistic"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def prepare_data(algo_name, X: np.ndarray) -> np.ndarray:
    """
    Some algorithms require the data in some special format.
    For example:
      - it is better to have the data in 'float64' instead than 'float32'
      - the data must be 'mean=0, standard_deviation=1'
      - etc
    :param algo_name: algo used
    :param X: dataset
    :return: the processed dataset
    """

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


def process_algorithms():
    for algo_name in GRAPH_ALGORITHMS:
        # skip algorithms staring with '#...'
        if algo_name.startswith("#"):
            continue

        # create the directory containing the results
        os.makedirs(f"data/{algo_name}", exist_ok=True)

        graphs_processed = list_graph_processed(algo_name)

        if N_JOBS <= 1:
            graphs_to_process = list_graph_names()
            sequential_process_graphs(algo_name, graphs_to_process, graphs_processed)
        else:
            graphs_to_process = list_graph_names(n_parts=N_PARTS)
            Parallel(n_jobs=N_JOBS)(
                delayed(parallel_process_graphs)(algo_name, i + 1, graphs_to_process[i], graphs_processed)
                for i in range(N_PARTS)
            )
    pass


def list_graph_names(n_parts: int = 1) -> Union[list[str], list[list[str]]]:
    """
    Collect all graphs names available in 'c'
    Note: the graphs are located in 'c' in increasing order or vertices' counts
        When used with 'Parallel', it is better to shuffle the names in such way
        more or less all Python instances will process the assigned graphs in
        similar time.
    """
    log.info('Collect graphs to process')

    with h5py.File(GRAPH_DATASETS, "r") as c:
        # collect the graph names
        graph_names = []
        for order in GRAPH_ORDERS:
            graphs_list = c[order]

            for graph_id in graphs_list.keys():
                graph_info = graphs_list[graph_id]
                n = graph_info.attrs['n']
                m = graph_info.attrs['m']

                # skip some graphs
                if n in EXCLUDE_GRAPH_DEGREES:
                    continue

                graph_names.append(graphs_list[graph_id].name)
            # end
        # end
    # end

    # sequential processing
    if n_parts <= 1:
        return graph_names

    # parallel processing: split the list in 'n_parts' sub-lists of
    # same size except the last one. The list is shuffled to be sure
    # that all processes will terminate in similar times
    rnd.shuffle(graph_names)
    part_len = (len(graph_names) + n_parts - 1) // n_parts

    graph_names_lists = []
    for i in range(n_parts - 1):
        s = i*part_len
        graph_names_lists.append(graph_names[s:s+part_len])
    s = (n_parts - 1) * part_len
    graph_names_lists.append(graph_names[s:])

    return graph_names_lists


def list_graph_processed(algo_name: Optional[str] = None) -> list[str]:
    processed = set()

    parent_dir = path("data") if algo_name is None else path(f"data/{algo_name}")
    for graph_predictions_name in parent_dir.files("graphs-predictions*.hdf5"):
        p = None
        try:
            p = h5py.File(graph_predictions_name, "r")

            for gorder in p.keys():
                graphs = p[gorder]
                for graph_id in graphs.keys():
                    name = graphs[graph_id].name
                    processed.add(name)

            p.close()
        except OSError as e:
            p = h5py.File(graph_predictions_name, "w")
            p.close()
    # end
    return list(processed)


# ---------------------------------------------------------------------------
# Graph processing
# ---------------------------------------------------------------------------

def process_graphs(n_jobs=0):
    # collect the list of graphs
    # can be executed in parallel

    graphs_processed = list_graph_processed()

    if n_jobs > 1:

        n_parts = n_jobs

        graphs_to_process = list_graph_names(n_parts=n_parts)
        Parallel(n_jobs=n_jobs)(delayed(parallel_process_graphs)(i + 1, graphs_to_process[i], graphs_processed)
                                for i in range(n_parts))
    else:
        graphs_to_process = list_graph_names()
        sequential_process_graphs(graphs_to_process, graphs_processed)
    return
# end


def sequential_process_graphs(algo_name: str, graphs_to_process: list[str], graphs_processed: list[str]):
    log.info(f"Process sequential started on {len(graphs_to_process)} graphs")
    c = h5py.File(GRAPH_DATASETS, "r")

    graph_predictions_name: str = \
        "data/graphs-predictions.hdf5" if algo_name is None else f"data/{algo_name}/graphs-predictions.hdf5"

    n_graphs = len(graphs_to_process)
    for i in range(n_graphs):
        graph_name = graphs_to_process[i]
        graph_info = c[graph_name]
        name = graph_info.name

        if name in graphs_processed:
            continue

        n = graph_info.attrs['n']
        m = graph_info.attrs['m']

        log.info(f"... {name}: {n} x {m} [{i+1}/{n_graphs}]")

        process_graph(algo_name, graph_info, graph_predictions_name)
        # break       # FILTER
    return
# end


def parallel_process_graphs(algo_name: str, process_id: int, graph_names: list[str], graphs_processed: list[str]):
    # Initialize the logging system in THIS process
    assert isinstance(algo_name, str)
    assert isinstance(process_id, int)
    assert is_instance(graph_names, list[str])
    assert is_instance(graphs_processed, list[str])

    global log
    logging.config.fileConfig("logging_config.ini")
    log = logging.getLogger(f"main.p{process_id:02}")
    log.info(f"Process {process_id:02} started on {len(graph_names)} graphs")

    # load the datasets
    c = h5py.File(GRAPH_DATASETS, "r")

    # initialize the container for the results
    graph_predictions_name = \
        f"data/graphs-predictions-{process_id:02}.hdf5" if algo_name is None else f"data/{algo_name}/graphs-predictions-{process_id:02}.hdf5"

    # scan the graphs
    n_graphs = len(graph_names)
    for i in range(n_graphs):
        graph_name = graph_names[i]
        graph_info = c[graph_name]
        name = graph_info.name

        if name in graphs_processed:
            continue

        n = graph_info.attrs["n"]
        m = graph_info.attrs["m"]

        log.info(f"... {name}: {n} x {m} [{i+1}/{n_graphs}]")

        process_graph(algo_name, graph_info, graph_predictions_name)
        # break
    # end

    log.info(f"Done")
    return
# end


def process_graph(algo_name: str, graph_info: h5py.Group, graph_predictions_name: str):
    assert isinstance(algo_name, str)
    assert isinstance(graph_info, h5py.Group)
    assert isinstance(graph_predictions_name, str)

    name = graph_info.name
    n = graph_info.attrs["n"]
    m = graph_info.attrs["m"]
    wl_hash = graph_info.attrs["wl_hash"]

    A = graph_info.attrs["adjacency_matrix"]

    pred_list = []

    # scan sem_type
    for sem_type in graph_info.keys():

        if sem_type in EXCLUDE_SEM_TYPES:
            continue

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

            try:
                algo.learn(X)
                C: np.ndarray = algo.causal_matrix.astype(np.int8)

                CM[i, :, :] = C
            except Exception as e:
                log.full_error(e, f"Unable to analyze the data {name}/{sem_type} using {algo_name}")
                CM[i, :, :] = 0

            # break   # FILTER
        # end
        # dset = p.create_dataset(f"{name}/{algo_name}/{sem_type}", (nds, n, n), dtype=CM.dtype, data=CM)
        # dset.attrs["method"] = method
        # dset.attrs["sem_type"] = sem_type
        # dset.attrs["algorithm"] = algo_name

        pred_list.append(dict(
            name=name,
            algo_name=algo_name,
            method=method,
            sem_type=sem_type,
            nds=nds,
            n=n,
            CM=CM
        ))
        # break       # FILTER
    # break           # FILTER

    p = h5py.File(graph_predictions_name, "a")
    pred_info = p.create_group(name)
    pred_info.attrs["n"] = n
    pred_info.attrs["m"] = m
    pred_info.attrs["adjacency_matrix"] = A
    pred_info.attrs["wl_hash"] = wl_hash

    for pred_dict in pred_list:
        name = pred_dict['name']
        algo_name = pred_dict['algo_name']
        method = pred_dict['method']
        sem_type = pred_dict['sem_type']
        nds = pred_dict['nds']
        CM = pred_dict['CM']

        dset = p.create_dataset(f"{name}/{algo_name}/{sem_type}", (nds, n, n), dtype=CM.dtype, data=CM)
        dset.attrs["method"] = method
        dset.attrs["sem_type"] = sem_type
        dset.attrs["algorithm"] = algo_name
    # end
    p.close()

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

    process_algorithms()

    log.info("done")
# end


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
