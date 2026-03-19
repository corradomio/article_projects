import logging.config
import random

import h5py
import numpy as np
from path import Path as path
import matplotlib.pylab as plt
import netx
from netx.draw import draw
import stdlib.iset as iset
import stdlib.jsonx as json
from stdlib.tprint import tprint
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
#   process_file
#       process_graph
# ---------------------------------------------------------------------------

def process_file(datafile: path):
    tprint(f"... {datafile.name}")

    graphs = json.load(datafile)["graphs"]
    for sorder in graphs:
        graph_list = graphs[sorder]
        total = len(graph_list)
        count = 0
        for ginfo in graph_list:
            plot_graph(ginfo, count, total)
            count += 1
    pass
# end


def plot_graph(ginfo: dict, c, t):
    n = ginfo["n"]
    m = ginfo["m"]
    wl_hash = ginfo["wl_hash"]
    W = np.array(ginfo["adjacency_matrix"])

    tprint(f"... {c}/{t}", force=False)

    plt.clf()
    draw(W, label=f"G=(|V|={n},|E|={m})")

    s = c//1000
    fpath = path(f"./plots/{n}/{(s+1):02}/{wl_hash}.png")
    fpath.parent.makedirs_p()

    plt.savefig(fpath, dpi=300)
# end


# ---------------------------------------------------------------------------
#
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

