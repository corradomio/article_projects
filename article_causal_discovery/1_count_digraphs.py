import os

import castle
import networkx as nx
import netx
import numpy as np
from path import Path as path

import stdlib.loggingx as logging
from stdlib.jsonx import load
from stdlib.tprint import tprint
import matplotlib.pyplot as plt



def main():
    log = logging.getLogger('main')
    log.info(f"nx: {nx.__version__}")
    log.info(f"castle: {castle.__version__}")

    os.makedirs('data_plots', exist_ok=True)

    root = path("data")
    for f in root.files("*.json"):
        # print(f)
        data = load(f)
        graphs = data["graphs"]

        for sorder in graphs.keys():
            n = int(sorder)
            # if n < 7: continue
            # if n > 7: continue

            count  = 0

            graphs_n = graphs[sorder]
            for g in graphs_n:
                wl_hash = g["wl_hash"]
                if "-" in wl_hash: continue

                count += 1
                # tprint(f"... {count}", force=False)
                # plt.show()
                pass
            # end for
            tprint(f"... {n} nodes: {count} graphs on {len(graphs_n)}")
        # end
    # end

    log.info('done')
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
