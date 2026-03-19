import os
import sys
from stdlib import logging
import multiprocessing
import traceback
from typing import Any, cast
from path import Path as path
import numpy as np
import h5py
import torch
from joblib import Parallel, delayed
from stdlib.qname import import_from, create_from
from stdlib import jsonx
from stdlib.dictx import dict_exclude
import warnings

warnings.filterwarnings("ignore")

logging.config.fileConfig("logging_config.ini")
LOG = logging.getLogger("main")
LOG.info("Logging initialized")

LOG.info(f"python {sys.version}")
LOG.info(f"torch {torch.__version__}")
LOG.info(f"numpy {np.__version__}")

DATASETS = path(r"../article_causal_discovery_bool_data/datasets")
RESULTS  = path(r"../article_causal_discovery_bool_data/results")

os.environ["CASTLE_BACKEND"] = "pytorch"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fullpath_of(dataset: str):
    ds_orig = dataset
    dataset = dataset.replace("\\", "/")
    if (dataset.find('/') == -1):
        dataset = f"{DATASETS}/{dataset}"
    if os.path.exists(dataset):
        return path(dataset)
    if not dataset.endswith(".hdf5"):
        dataset += ".hdf5"
    if os.path.exists(dataset):
        return path(dataset)
    # raise FileNotFoundError(f"{ds_orig} not found")
    raise FileNotFoundError(f"The path '{ds_orig}' is  NOT a valid HDF5 file or directory containing HDF5 files")
# end


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


def resultof(file: path, algo: str, index: int) -> path:
    #  12345678
    # "dataset-<order>.hdf5"
    # "dataset-<order>-sampled-<index>.hdf5"
    # -> "<algo>-<order>.hdf5"
    #    "<algo>-<order>-<index>.hdf5"
    suffix = file.stem[8:]
    name = f"{algo}-{suffix}-{index:02}.hdf5"
    return RESULTS / name
# end


PARAMS_TO_EXCLUDE = ["class", "comment", "note", "skip"]


# ---------------------------------------------------------------------------
# algorithms
# ---------------------------------------------------------------------------

def apply_algo(ginfo, algo, jid):
    aname = algo.__class__.__name__

    wl_hash = ginfo.attrs["wl_hash"]
    n = ginfo.attrs["n"]
    # m = ginfo.attrs["m"]
    # fun = ginfo.attrs["fun"]
    adjacency_matrix: np.ndarray = ginfo.attrs["adjacency_matrix"].astype(np.int8)  # (<n>, <n>)

    # (<n_datasets>,<n_record>,<n_columns>) : (100, 3000, <n>)
    dataset: np.ndarray = np.array(ginfo["dataset"])

    k = dataset.shape[0]        # 100
    assert n == len(adjacency_matrix)

    # create the tensor containing the causal matrices
    # set as FIRST matrix the current adjacency_matrix (GROUND TRUTH)
    causal_matrices = np.zeros((k+1, n, n), dtype=np.int8)
    causal_matrices[0,...] = adjacency_matrix

    for i in range(k):
        LOG.infot(f"graph: /{n}/{wl_hash} ... {i+1}/{k}")
        try:
            X: np.ndarray = dataset[i]

            # convert into a float
            X = X.astype(np.float64)

            algo.learn(X)
            causal_matrix: np.ndarray = algo.causal_matrix

            if causal_matrix.sum() == 0:
                LOG.error(f"graph: /{n}/{wl_hash}/{i} -> no causal matrix")

            causal_matrices[i+1,...] = causal_matrix
        except Exception as e:
            exc = traceback.format_exc()
            LOG.error(f"{aname}: /{n}/{wl_hash} -> {e}\n{exc}")
            pass

    return causal_matrices
# end

def save_causal_matrices(dst, k, ginfo, causal_matrices):
    cinfo = dst.create_group(k)
    cinfo.attrs["n"] = ginfo.attrs["n"]
    cinfo.attrs["m"] = ginfo.attrs["m"]
    cinfo.attrs["wl_hash"] = ginfo.attrs["wl_hash"]
    cinfo.attrs["fun"] = ginfo.attrs["fun"]
    cinfo.create_dataset("causal_matrices",
                         causal_matrices.shape,
                         dtype=causal_matrices.dtype,
                         data=causal_matrices)
# end


# ---------------------------------------------------------------------------
# algorithms
# ---------------------------------------------------------------------------

def process_keys_par(source: path, jid: int, gkeys: list[str], aname: str, ainfo: dict):
    global LOG
    logging.config.fileConfig("logging_config.ini")
    LOG = logging.getLogger(f"{jid:02}.{aname}")

    process_keys(source, jid, gkeys, aname, ainfo)
# end


def process_keys(source: path, jid: int, gkeys: list[str], aname: str, ainfo: dict, max_graphs=-1):

    # name of the file containing the results
    result = resultof(source, aname, jid)

    # create the algorithm
    # algo = create_algo(ainfo)
    algo = create_from(ainfo)

    aname = algo.__class__.__name__
    LOG.info(f"file: {source}")

    # source & destination HDF5 files
    src = h5py.File(source, mode='r')
    # initialize the file
    dst = h5py.File(result, mode='w')
    dst.close()

    # scan the selected keys
    index = 0
    total = len(gkeys)
    for k in gkeys:
        index += 1
        LOG.info(f"graph: {k} ({index}/{total})")

        # retrieve the graph information
        ginfo = src[k]
        # adjacency_matrix: matrix(n x n) or int32
        # fun: string
        # n: int32
        # m: int32
        # wl_hash: string

        # apply the algorithm
        causal_matrices = apply_algo(ginfo, algo, jid)

        # save the inferred causal matrices
        dst = h5py.File(result, mode='a')
        save_causal_matrices(dst, k, ginfo, causal_matrices)
        dst.close()

        if max_graphs > 0 and index >= max_graphs:
            break
    # end

    # dst.close()
    src.close()
# end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# N_JOBS = "auto"
N_JOBS = 10

def main(argv):
    # load the algorithm
    # parameters: [_, algo, dataset, dir_results]
    # 'algo' must be ('*' or 'all') OR an algo defined in 'algorithms.json'
    # if dataset is a directory, process ALL files in the directory
    # 'results' MUST BE a directory

    algorithms = jsonx.load("algorithms.json")
    algo_list = list(algorithms.keys())
    files_to_process = DATASETS.files("*.hdf5")

    # [_, algo, ..]
    if len(argv) >= 2:
        algo = argv[1]
        if algo in ["*", "all"]:
            algo_list = list(algorithms.keys())
        elif algo not in algo_list:
            raise ValueError(f"Algorithm '{algo}' not present in 'algorithms.json'")
        else:
            algo_list = [algo]

    # [_, algo, dataset, ..]
    if len(argv) >= 3:
        dataset = fullpath_of(argv[2])
        if dataset.is_file():
            files_to_process = [dataset]
        elif dataset.is_dir():
            files_to_process = dataset.files("*.hdf5")
        else:
            raise ValueError(f"The path '{argv[3]}' is  NOT a valid HDF5 file or directory containing HDF5 files")

    # [_, algo, dataset, dir_results]
    if len(argv) >= 4:
        dir_results = path(argv[3])
        if dir_results.is_file():
            raise ValueError(f"The path '{argv[4]}' is NOT a valid directory")
        elif dir_results.is_dir():
            pass
        elif not dir_results.exists():
            dir_results.mkdir_p()

        global RESULTS
        RESULTS = dir_results
    # end

    for aname in algo_list:
        # skip algos starting with '#'
        if cast(str, aname).startswith("#"):
            continue

        LOG.info(f"Algorithm '{aname}'")
        ainfo = algorithms[aname]

        for file in files_to_process:
            LOG.info(f"... Processing '{file.stem}'")
            # if "-4-" not in file.stem: continue
            gkeys = keysof(file)
            kparts = split(gkeys, N_JOBS)

            if N_JOBS <= 0:
                for i, kpart in enumerate(kparts):
                    process_keys(file, i, kpart, aname, ainfo, 1)
                    break
            else:
                Parallel(n_jobs=N_JOBS)(delayed(process_keys_par)(file, i, kpart, aname, ainfo)
                                        for i, kpart in enumerate(kparts))
            pass
        # end
    LOG.info(f"Done")
# end


if __name__ == "__main__":
    main(sys.argv)
