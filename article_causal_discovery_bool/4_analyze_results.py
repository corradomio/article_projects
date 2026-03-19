import csv
import sys
import warnings

import h5py
import networkx as nx
import numpy as np
import torch
from h5py import Dataset
from path import Path as path

import netx
from stdlib import logging
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

logging.config.fileConfig("logging_config.ini")
LOG = logging.getLogger("main")
LOG.info("Logging initialized")

LOG.info(f"python {sys.version}")
LOG.info(f"torch {torch.__version__}")
LOG.info(f"numpy {np.__version__}")

DATASETS = path(r"../article_causal_discovery_bool_data/datasets")
RESULTS  = path(r"../article_causal_discovery_bool_data/results")
MERGED   = path(r"../article_causal_discovery_bool_data/merged")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def algo_of(name: str) -> str:
    p = name.index('-')
    return name[:p]

# ---------------------------------------------------------------------------
# analyze_result
# ---------------------------------------------------------------------------

def _clear_diagonal(A: np.ndarray) -> np.ndarray:
    n, _ = A.shape
    for i in range(n):
        A[i,i] = 0
    return A


def analyze_result_par(r_file: path) -> list:
    global LOG
    logging.config.fileConfig("logging_config.ini")
    LOG = logging.getLogger("main")
    return analyze_result(r_file)
# end


def analyze_result(r_file: path) -> list:
    LOG.info(f"Analyzing {r_file.stem}")
    r = h5py.File(r_file, mode='r')

    analysis = []
    # scan graph order
    algo = algo_of(r_file.stem)
    for order in r.keys():
        n = int(order)
        max_size = n*(n-1)

        n_graphs = 0
        n_datasets = 0

        n_empty_graphs = 0
        n_full_graphs = 0
        n_partial_graphs = 0
        n_dags = 0
        n_direct_graphs = 0
        n_not_connected = 0
        max_edges = 0
        max_uedges = 0

        r_order = r[order]
        gids = r_order.keys()
        ngids = len(gids)

        for i, gid in enumerate(gids):
            n_graphs += 1

            LOG.tprint(f"... graph: /{order}/{gid} [{i+1:2}/{ngids}]", force=False)
            ginfo = r_order[gid]
            gdata: Dataset = ginfo["causal_matrices"]
            nk = gdata.shape[0]

            adjacency_matrix = gdata[0, :, :]
            # G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
            # nx.draw(G, pos=nx.circular_layout(G))
            # plt.show()

            for k in range(1, nk):
                n_datasets += 1

                causal_matrix = gdata[k, :, :]
                is_empty = netx.is_empty_adjacency_matrix(causal_matrix)
                is_full  = netx.is_full_adjacency_matrix(causal_matrix)

                if causal_matrix.sum() > max_size:
                    LOG.tprint(f"... {r_file.stem}:/{order}/{gid} has an invalid adjacency matrix")
                    _clear_diagonal(causal_matrix)

                H = nx.from_numpy_array(causal_matrix, create_using=nx.DiGraph)
                n_edges = H.size()

                is_partial = netx.is_partial_adjacency_matrix(causal_matrix)
                is_dag = netx.is_directed_acyclic_graph(H)
                is_digraph = not is_empty and not is_partial and not is_dag
                is_not_connected = not nx.is_weakly_connected(H)
                n_uedges = len(netx.partial_graph_undirected_edges(H))

                assert is_partial == netx.is_partial_directed_acyclic_graph(H)
                assert 1 == (is_empty + is_partial + is_dag + is_digraph)

                n_empty_graphs   += is_empty
                n_full_graphs    += is_full
                n_partial_graphs += is_partial
                n_dags += is_dag
                n_direct_graphs += is_digraph
                n_not_connected += is_not_connected
                max_edges  = max(max_edges,  n_edges)
                max_uedges = max(max_uedges, n_uedges)
                pass
            pass

        n_total_graphs = (n_empty_graphs + n_partial_graphs + n_dags + n_direct_graphs) - n_empty_graphs
        analysis.append([
            algo, order, n_graphs, n_datasets,
            max_edges, max_uedges,
            n_not_connected,
            n_empty_graphs, n_full_graphs,
            n_partial_graphs, n_dags, n_direct_graphs,
            n_total_graphs
        ])
    assert len(analysis) == 1
    return analysis[0]
    pass
# end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_JOBS = 14
# N_JOBS = 0

def main(argv: list[str]):
    header = [[
        "algo", "order", "n_graphs", "n_datasets",
        "max_edges", "max_uedges",
        "n_not_connected",
        "n_empty_graphs", "n_full_graphs",
        "n_partial_graphs", "n_dags", "n_direct_graphs",
        "n_total_graphs"
    ]]

    if N_JOBS == 0:
        analysis = []
        for r in MERGED.walkfiles("*.hdf5"):
            # if "notearsnonlinear" not in r: continue
            ar = analyze_result(r)
            analysis += ar
    else:
        analysis = Parallel(n_jobs=N_JOBS)(
            delayed(analyze_result_par)(r)
            for r in MERGED.walkfiles("*.hdf5")
        )

    analysis = header + analysis
    with open("analysis.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(analysis)
    pass


if __name__ == "__main__":
    main(sys.argv)
