import sys
import warnings

import h5py
import networkx as nx
import numpy as np
from h5py import Dataset
from joblib import Parallel, delayed
from path import Path as path

import netx
import netx.metrics
from stdlib import logging

warnings.filterwarnings("ignore")

logging.config.fileConfig("logging_config.ini")
LOG = logging.getLogger("main")
LOG.info("Logging initialized")

LOG.info(f"python {sys.version}")
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
# Utilities
# ---------------------------------------------------------------------------

def algo_of(name: str) -> str:
    p = name.index('-')
    return name[:p]


def order_of(name: str) -> int:
    p = name.index('-')
    return int(name[p+1:])


def keysof(file: path) -> list[str]:
    f = h5py.File(file, mode="r")
    keys = []
    for order in f.keys():
        graphs = f[order]
        keys = [
            f"/{order}/{k}"
            for k in graphs.keys()
        ]
    f.close()
    return keys
# end


def split(keys, n:int) -> list[list[str]]:
    n = max(1, n)
    m = len(keys)
    size = (len(keys) + n-1)//n
    parts = []
    for p in range(0, m, size):
        e = min(p+size, m)
        parts.append(keys[p:e])
    # end
    return parts
# end

# ---------------------------------------------------------------------------
# algorithms
# ---------------------------------------------------------------------------


def process_keys_par(causal_path: path, jid, keys, aname, order):
    global LOG
    logging.config.fileConfig("logging_config.ini")
    LOG = logging.getLogger(f"{jid:02}.{aname}/{order}")
    process_keys(causal_path, jid, keys, aname, order)
    pass
# end


def process_keys(causal_path: path, jid, keys, aname, order):
    if "notearsnonlinear" in causal_path:
        return

    ngids = len(keys)

    metrics_path = RESULTS / f"{causal_path.stem}-metrics-{jid:02}.hdf5"

    causal_file  = h5py.File(causal_path,  mode='r')
    metrics_file = h5py.File(metrics_path, mode='w')

    for g, gid in enumerate(keys):
        ginfo = causal_file[gid]
        causal_matrices: Dataset = ginfo["causal_matrices"]

        # n_instances, nk, nk
        # Note:
        #   causal_matrices[0,:,:] is the ground truth
        #
        nk, n, _ = causal_matrices.shape

        # n_metrics, nk, nk
        metrics = ["SHD", "SID", "SSID", "DSD"]
        n_metrics = len(metrics)       # SID, SHD
        metrics = np.zeros((n_metrics, nk,nk), dtype=float)

        ij = 0
        nij = nk*(nk-1)//2
        for i in range(nk):
            Ai = causal_matrices[i, :, :]
            Gi = nx.from_numpy_array(Ai, create_using=nx.DiGraph)
            for j in range(i+1, nk):
                ij += 1
                LOG.tprint(f"... ... {gid} [{ij:3}/{nij}, {g + 1:4}/{ngids}]", force=False)

                Aj = causal_matrices[j, :, :]
                Gj = nx.from_numpy_array(Aj, create_using=nx.DiGraph)

                assert Gi.order() == Gj.order()

                SHD = netx.metrics.structural_hamming_distance_pdag(Ai, Aj)
                SID = netx.metrics.structural_intervention_distance(Ai, Aj)
                SSID = netx.metrics.symmetric_intervention_distance_pdag(Ai, Aj)
                DSD = netx.metrics.d_separation_distance_pdag(Gi, Gj)

                metrics[0, i, j] = SHD
                metrics[1, i, j] = SID
                metrics[2, i, j] = SSID
                metrics[3, i, j] = DSD
                # break
            # end
            # break
        # end

        # save the metrics
        metrics_file[gid + "/shd"] = metrics[0]
        metrics_file[gid + "/sid"] = metrics[1]
        metrics_file[gid + "/ssid"] = metrics[2]
        metrics_file[gid + "/dsd"] = metrics[3]

        metrics_gid = metrics_file[gid]
        # copy the attributes
        for a in ginfo.attrs:
            metrics_gid.attrs[a]= ginfo.attrs[a]
        pass

    metrics_file.close()
    pass
# end


# N_JOBS = 14
N_JOBS = 0

def analyze_result(file: path):
    aname = algo_of(file.stem)
    order = order_of(file.stem)

    LOG.info(f"... Processing '{file.stem}'")

    gkeys = keysof(file)
    kparts = split(gkeys, N_JOBS)

    if N_JOBS == 0:
        for i, kpart in enumerate(kparts):
            process_keys(file, i, kpart, aname, order)
            break
    else:
        Parallel(n_jobs=N_JOBS)(
            delayed(process_keys_par)(file, i, kpart, aname, order)
            for i, kpart in enumerate(kparts)
        )
    pass



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str]):
    for r in MERGED.walkfiles("*.hdf5"):
        analyze_result(r)
    pass
# end

if __name__ == "__main__":
    main(sys.argv)
