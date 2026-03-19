import stdlib.logging as logging
import random

import h5py
import numpy as np
from joblib import Parallel, delayed
from path import Path as path

import netx
import stdlib.iset as iset
import stdlib.jsonx as json


# ---------------------------------------------------------------------------
#   process_file
#       process_graph
# ---------------------------------------------------------------------------

LOG = None

GRAPHS_DIR = path(r"..\article_causal_discovery_bool_data\graphs")
DATASETS_DIR = path(r"..\article_causal_discovery_bool_data\datasets")

MAX_BINOMIAL_PROB = 0.3
MAX_GRAPHS = 3000       # n max of graphs for each order
N_INSTANCES = 30        # n of datasets for each graph
N_ENTRIES = 4000        # n of records for each dataset. Note: 9->8000, 10->16000

N_ELEMENTS = 20000      # n of graphs in each HDF5 container


def vars(nodes: list[int]) -> list[str]:
    return [f"x{n}" for n in nodes]


def orderof(datafile) -> str:
    suffix: str = datafile.stem[12:]
    if suffix.endswith("-sampled"):
        suffix = suffix[:-8]
    return suffix


def select_graphs(graph_list: list[dict], n_graphs: int) -> list[dict]:
    if n_graphs > 0 and len(graph_list) <= n_graphs:
        return graph_list
    elif n_graphs > 0:
        return random.sample(graph_list, n_graphs)
    else:
        return graph_list


def process_file(datafile: path):
    container_name = orderof(datafile)
    log_suffix = container_name if container_name  == "10" else ("0"+ container_name)

    global LOG
    logging.config.fileConfig("logging_config.ini")
    LOG = logging.getLogger(f"order.{log_suffix}")
    LOG.info(f"Processing {datafile}")

    icontainer = 0

    container_file = DATASETS_DIR / f"dataset-{container_name}-{icontainer}.hdf5"
    container_file.remove_p()
    container = h5py.File(container_file, "w")

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

        graph_list: list[dict] = graphs[sorder]
        graph_list = select_graphs(graph_list, MAX_GRAPHS)
        total = len(graph_list)

        for c, ginfo in enumerate(graph_list):
            count = c+1
            LOG.infot(f"[{sorder:2}] {c:4}/{total}")

            if count//N_ELEMENTS != icontainer:
                container.close()

                icontainer = count//N_ELEMENTS

                container_file = path(f"datasets/dataset-{container_name}-{icontainer}.hdf5")
                container_file.remove_p()
                container = h5py.File(container_file, "w")
            # end

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

            LOG.infot(f"[{count:4}/{total:4}] {wl_hash}: ({n},{m})")

            dataset, finfos = generate_dataset(ginfo)

            # function information in json format
            jfun = json.dumps(finfos, separators=(',', ':'))

            grp = container.create_group(f"{n}/{wl_hash}")
            grp.attrs["n"] = n
            grp.attrs["m"] = m
            grp.attrs["wl_hash"] = wl_hash
            grp.attrs["adjacency_matrix"] = W
            grp.attrs["fun"] = jfun

            container.create_dataset(
                f"{n}/{wl_hash}/dataset",
                dataset.shape,
                dtype=dataset.dtype,
                data=dataset)
        # end

        container.close()

        LOG.info(f"[order={sorder}, total={total}] done")
    pass
# end


def generate_dataset(graph: dict) -> tuple[np.ndarray, list]:
    # logger = logging.getLogger("process_graph")
    n = graph["n"]
    m = graph["m"]
    wl_hash = graph["wl_hash"]
    adjacency_matrix = np.array(graph["adjacency_matrix"])

    G = netx.from_numpy_matrix(adjacency_matrix)

    assert n == G.order()
    assert m == G.size()

    dataset_list: list[np.ndarray] = []
    finfo_list: list[dict] = []

    for i in range(N_INSTANCES):
        dataset, fun_info = generate_instance(i, wl_hash, G)

        dataset_list.append(dataset)
        finfo_list.append(fun_info)
    # end

    datase_array = np.array(dataset_list)
    # print(datase_array.shape)
    return datase_array, finfo_list
# end


def generate_instance(i: int, wl_hash: str, G: netx.Graph) -> tuple[np.ndarray, dict]:
    # to generate:
    #   1) the dataset
    #   2) the
    # boolean information

    fun_info = {
        "instance": i,
        "n": G.order(),
        "m": G.size(),
        "wl_hash": wl_hash,
        "nodes": {}
    }

    fun_nodes = fun_info["nodes"]

    # n of nodes
    order = G.order()

    # expression vars

    # set of already processed nodes
    processed: set[int] = set()

    n_entries = N_ENTRIES
    if order == 9:
        n_entries *= 2
    if order == 10:
        n_entries *= 4

    # numpy array to fill
    dataset = np.zeros((n_entries, order), dtype=np.byte)

    # source nodes: nodes with in-degree = 0
    sources: set[int] = netx.sources(G)

    # initialize all source nodes with random values
    # note: it is not necessary to add a binomial noise
    for s in sources:
        bool_fun: np.ndarray = np.random.binomial(1, 0.5, n_entries)
        dataset[:, s] = bool_fun

        fun_nodes[s] = {
            "n": s,
            # "f": [0, 1],
            "f": f"x{s}",
            "params": [],
            "noisep": 0.5,
        }

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

            # generate a random probability for the binomial noise
            noise_prob = random.uniform(0., MAX_BINOMIAL_PROB)

            # generate the random noise
            bin_noise = np.random.binomial(1, noise_prob, size=n_entries)

            # generate a random boolean function with n parameters
            # excluding constant functions (0, 2**n-1)
            bool_fun, bool_expr = iset.ibooltable(-1, n_params, allvars=True, vars=vars(predecessors))
            fx = ("%X" % iset.ibinset(bool_fun))
            # nx = ("%X" % iset.ibinset(bin_noise))

            fun_nodes[n] = {
                "n": n,
                # "f": bool_fun,
                "f": bool_expr,
                "params": predecessors,
                "noisep": noise_prob,
                "fx": fx,
                # "nx": nx
            }

            # efficient evaluation of the boolean function
            bool_indices = to_iset(dataset[:, predecessors])
            bool_apply = np.vectorize(lambda i: bool_fun[i])

            # apply the function and add the noise (using xor)
            dataset[:, n] = bool_apply(bool_indices) ^ bin_noise

            # update the list of processed nodes
            processed.update(to_process)
        pass
    # end

    return dataset, fun_info
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


def process_file_(datafile: path):
    print(datafile)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_JOBS = 7

def main():
    logging.getLogger("main").info("start processing ...")
    datafiles = GRAPHS_DIR.files("*.json")

    # for datafile in datafiles:
    #     process_file(datafile)
    Parallel(n_jobs=N_JOBS)(delayed(process_file)(datafile) for datafile in datafiles)

    logging.getLogger("process_file").info("done")
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
