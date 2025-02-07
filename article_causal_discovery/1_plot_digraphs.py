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
        print(f)
        data = load(f)
        graphs = data["graphs"]

        count  = 0
        for sorder in graphs.keys():
            n = int(sorder)
            if n < 7: continue
            if n > 7: continue

            graphs_n = graphs[sorder]
            for g in graphs_n:
                wl_hash = g["wl_hash"]
                if "-" in wl_hash: continue

                if n <= 5:
                    pdir = f"data_plots/{n}"
                    os.makedirs(pdir, exist_ok=True)
                    fname = f"{pdir}/{wl_hash}.png"
                else:
                    m = g["m"]
                    pdir = f"data_plots/{n}/{m}"
                    os.makedirs(pdir, exist_ok=True)
                    fname = f"{pdir}/{wl_hash}.png"
                # end

                if os.path.exists(fname): continue

                adjmat = np.array(g["adjacency_matrix"])

                G = nx.from_numpy_array(adjmat, create_using=nx.DiGraph)

                if count% 50 == 0:
                    plt.close()
                plt.clf()

                netx.draw(G)
                plt.savefig(fname, dpi=300)

                count += 1
                tprint(f"... {count}", force=False)
                # plt.show()
                pass
            # end for
        # end
    # end

    log.info('done')
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
