import sys
import warnings

import h5py
import numpy as np
import torch
from h5py import Dataset, AttributeManager, Group
from path import Path as path

from stdlib import logging
import networkx as nx

warnings.filterwarnings("ignore")

logging.config.fileConfig("logging_config.ini")
LOG = logging.getLogger("main")
LOG.info("Logging initialized")

LOG.info(f"python {sys.version}")
LOG.info(f"torch {torch.__version__}")
LOG.info(f"numpy {np.__version__}")

DATASETS = path(r"../article_causal_discovery_bool_data/datasets")
RESULTS  = path(r"../article_causal_discovery_bool_data/done")
MERGED   = path(r"../article_causal_discovery_bool_data/merged")


MERGED.makedirs_p()

# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------
# file, group, dataset, attributes

def parse_name(name: str) -> tuple[str, str, str]:
    # -> algo, order, thread_id
    parts = name.split("-")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    else:
        return parts[0], parts[1], "0"


def order_of(name: str) -> str:
    _, order, _ = parse_name(name)
    return order


def create_merged_file(algo: str, order: str) -> path:
    merged_file = MERGED / f"{algo}-{order}.hdf5"
    # merged_file = MERGED / f"{algo}.hdf5"
    if not merged_file.exists():
        m = h5py.File(merged_file, mode='w')
        m.create_group(order)
        m.close()
    return merged_file


def merge_results(r_file: path):
    print(f"Merging {r_file.stem}")
    algo, order, tid = parse_name(r_file.stem)

    merged_file = create_merged_file(algo, order)
    merged = h5py.File(merged_file, mode='a')
    result = h5py.File(r_file, mode='r')

    for g in result.keys():
        if not g in merged.keys():
            merged.create_group(g)

        rg = result[g]
        mg = merged[g]
        for gid in rg.keys():
            mg.create_group(gid)

            rggid: Group = rg[gid]
            causal_matrices: Dataset = rggid["causal_matrices"]
            attrs: AttributeManager = rggid.attrs

            mggid = mg[gid]
            mggid.create_dataset("causal_matrices", data=causal_matrices)
            for a in attrs:
                v = attrs[a]
                mggid.attrs[a] = v
            # end
        # end
    # end
    merged.close()
    result.close()
    pass
# end


def merge_datasets(merged_file: path, dataset_file: path):
    print(f"Merging {dataset_file.stem} / {merged_file.stem}")
    _, order, _ = parse_name(dataset_file.stem)

    dataset = h5py.File(dataset_file, mode='r')
    merged = h5py.File(merged_file, mode='a')

    for g in dataset.keys():

        dg = dataset[g]
        mg = merged[g]

        for gid in dg.keys():

            dggid: Group = dg[gid]
            mggid: Group = mg[gid]

            fun = dggid.attrs["fun"]
            mggid.attrs["fun"] = fun
        # end
    # end
    merged.close()
    dataset.close()

def main(argv: list[str]):
    for m in MERGED.walkfiles("*.hdf5"):
        m.remove()

    for r_file in RESULTS.walkfiles("*.hdf5"):
        merge_results(r_file)

    for dataset_file in DATASETS.walkfiles("*.hdf5"):
        order = order_of(dataset_file.stem)
        for merged_file in MERGED.walkfiles(f"*-{order}.hdf5"):
            merge_datasets(merged_file, dataset_file)
    pass


if __name__ == "__main__":
    main(sys.argv)
