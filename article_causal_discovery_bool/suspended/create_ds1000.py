from stdlib import logging
import traceback
from typing import Any, cast
from random import sample
from path import Path as path
import numpy as np
import h5py
from joblib import Parallel, delayed
from stdlib import import_from
from stdlib import jsonx
from stdlib import picklex
from stdlib.tprint import tprint
from stdlib.dictx import dict_exclude
import warnings


ROOT = path(r"../article_causal_discovery_bool_data/datasets_1000")


def finfoof(file: path) -> path:
    parent = file.parent
    suffix = file.stem[8:]
    return parent / f"finfos-{suffix}.pickle"


def outof(file: path) -> path:
    parent = file.parent
    stem = file.stem[8:]
    return parent / f"ds1000-{stem}.hdf5"


def process_file(file: path):
    di_file = file
    fi_file = finfoof(file)
    do_file = outof(file)

    tprint(di_file.stem)
    # tprint(fi_file.stem)

    di = h5py.File(di_file, mode='r')
    do = h5py.File(do_file, mode='w')
    fi = picklex.load(fi_file)

    for order in di.keys():
        di_order = di[order]
        fi_order = fi[order]

        do_order = do.create_group(order)

        keys = list(di_order.keys())
        keys = keys if len(keys) < 1000 else sample(keys, 1000)

        for k in keys:
            tprint(f"... {k}", force=False)

            gi = di_order[k]
            go = do_order.create_group(k)
            fun = fi_order[k]

            go.attrs["wl_hash"] = gi.attrs["wl_hash"]
            go.attrs["n"] = gi.attrs["n"]
            go.attrs["m"] = gi.attrs["m"]
            go.attrs["adjacency_matrix"] = gi.attrs["adjacency_matrix"]
            go.attrs["fun"] = jsonx.dumps(fun)

            ds = np.array(gi["dataset"])
            ds = ds[:20]

            go.create_dataset("dataset", shape=ds.shape, dtype=ds.dtype, data=ds)
            pass
        pass
    pass

    di.close()
    do.close()
    pass


def main():
    for file in ROOT.files("dataset-*.hdf5"):
        process_file(file)
    tprint("done")
# end


if __name__ == "__main__":
    main()
