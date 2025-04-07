import logging.config
import random

import h5py
import numpy as np
from path import Path as path

import netx
import stdlib.iset as iset
import stdlib.jsonx as json
from stdlib.tprint import tprint
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
#   process_file
#       process_graph
#
# ---------------------------------------------------------------------------

N_INSTANCES = 100
N_ENTRIES = 3000
MAX_BINOMIAL_PROB = 0.3


def process_file(datafile: path):
    # logger = logging.getLogger("process_file")
    # logger.info(f"... {datafile.name}")
    tprint(f"... {datafile.name}")
    container_name = datafile.stem[12:]

    #   {
    #       "graph": {
    #           "<order>": [
    #               {
    #                   "n": 2,
    #                   "m": 1,
    #                   "wl_hash": "8a8a4905c9b0f8fd7847dc408134a288",
    #                   "adjacency_matrix": [
    #                       [0,1],
    #                       [0,0]
    #                   ]
    #               },
    #               ...
    #           ]
    #       }
    #   }

    graphs = json.load(datafile)["graphs"]
    for sorder in graphs:
        container = h5py.File(f"datasets/dataset-{container_name}.hdf5", "w")

        graph_list = graphs[sorder]
        count = 0
        total = len(graph_list)
        for ginfo in graph_list:
            #   {
            #       "n": 2,
            #       "m": 1,
            #       "wl_hash": "8a8a4905c9b0f8fd7847dc408134a288",
            #       "adjacency_matrix": [ ... ]
            #   },
            n = ginfo["n"]
            m = ginfo["m"]
            wl_hash = ginfo["wl_hash"]
            W = ginfo["adjacency_matrix"]
            # if '-' in wl_hash:
            #     # logger.warning(f"... ... {wl_hash} skipped")
            #     continue

            dataset = generate_dataset(ginfo)

            grp = container.create_group(f"{n}/{wl_hash}")
            grp.attrs['n'] = n
            grp.attrs['m'] = m
            grp.attrs['wl_hash'] = wl_hash
            grp.attrs['adjacency_matrix'] = W

            dset = container.create_dataset(
                f"{n}/{wl_hash}/dataset", dataset.shape,
                dtype=dataset.dtype,
                data=dataset)

            count += 1
            tprint(f"[{count:4}/{total}] {wl_hash}: ({n},{m})", force=False)
        # end
        container.close()
        tprint(f"[{count:4}/{total}] done")
    pass
# end


def generate_dataset(graph: dict) -> np.ndarray:
    # logger = logging.getLogger("process_graph")
    n = graph["n"]
    m = graph["m"]
    wl_hash = graph["wl_hash"]
    adjacency_matrix = np.array(graph["adjacency_matrix"])

    G = netx.from_numpy_matrix(adjacency_matrix)

    assert n == G.order()
    assert m == G.size()

    # logger.info(f"... ... {wl_hash}: {G}")

    dataset_list: list[np.ndarray] = []
    # dataset_info_list: list[dict] = []

    for i in range(N_INSTANCES):
        # dataset, dataset_info = generate_instance(wl_hash, G)
        dataset = generate_instance(wl_hash, G)

        dataset_list.append(dataset)
        # dataset_info_list.append(dataset_info)
    # end

    datase_array = np.array(dataset_list)
    # print(datase_array.shape)
    return datase_array
# end


def generate_instance(wl_hash: str, G: netx.Graph) -> (np.ndarray, dict):
    # boolean informations
    # dataset_info = {}

    # n of nodes
    order = G.order()

    # set of already processed nodes
    processed: set[int] = set()

    # numpy array to fill
    dataset = np.zeros((N_ENTRIES, order), dtype=np.byte)

    # source nodes: nodes with in-degree = 0
    sources: set[int] = netx.sources(G)

    # initialize all source nodes with random values
    # note: it is not necessary to add a binomial noise
    for s in sources:
        bool_fun: np.ndarray = np.random.binomial(1, 0.5, N_ENTRIES)
        dataset[:, s] = bool_fun

    # register the processed node
    processed.update(sources)

    # process all remaining nodes
    while len(processed) != order:
        # list of nodes having as predecessors the already processed nodes
        successors: set[int] = netx.all_descendants(G, processed)
        # list of nodes not yet processed
        to_process: set[int] = successors - processed

        # process the nodes not yet processed
        for n in to_process:
            # predecessor nodes pointing to this node
            predecessors = list(netx.predecessors(G, n))
            # n or predecessors = n of parameters of the boolean function
            n_params: int = len(predecessors)

            # generate a random propability for the binomial noise
            noise_prob = random.uniform(0., MAX_BINOMIAL_PROB)

            # generate the random noise
            bin_noise = np.random.binomial(1, noise_prob, size=N_ENTRIES)

            # generate a random boolean function with n parameters
            bool_fun: list[int] = iset.ibooltable(-1, n_params)

            # efficient evaluation of the boolean function
            bool_indices = to_iset(dataset[:, predecessors])
            bool_apply = np.vectorize(lambda i: bool_fun[i])

            # apply the function and add the noise (using xor)
            dataset[:, n] = bool_apply(bool_indices) ^ bin_noise

            # process each entry in the datase
            # for i in range(N_ENTRIES):
            #     bool_eval: int = _eval_bool_fun(bool_fun, list(dataset[i, predecessors]))
            #     bin_noise = np.random.binomial(1, noise_prob)
            #     dataset[i, n] = bool_eval ^ bin_noise
            # pass

            # update the list of processed nodes
            processed.update(to_process)
        pass
    # end
    # dataset complete
    # return dataset, dataset_info
    return dataset
# end


def to_iset(array: np.ndarray) -> np.ndarray:
    n, m = array.shape
    S = np.zeros(n, dtype=int)
    F = 1
    for i in range(m):
        S += F*array[:, i]
        F *= 2
    return S
# end


# def _eval_bool_fun(fun: list[int], params: list[int]) -> int:
#     S = iset.ilistset(params)
#     return fun[S]
# # end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.getLogger("process_file").info("start processing ...")
    datafiles = path("data").files("*.json")

    # for datafile in datafiles:
    #     process_file(datafile)
    Parallel(n_jobs=10)(delayed(process_file)(datafile) for datafile in datafiles)

    logging.getLogger("process_file").info("done")
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
