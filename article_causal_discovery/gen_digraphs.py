import os
from datetime import datetime

import random as rnd
import castle
import networkx as nx

import netx
import stdlib.jsonx as jsx
import stdlib.loggingx as logging


def iround(r: float) -> int:
    return int(r+0.5)


def main():
    log = logging.getLogger('main')
    log.info(f"nx: {nx.__version__}")
    log.info(f"ig: {ig.__version__}")
    log.info(f"castle: {castle.__version__}")

    os.makedirs('data', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)

    n_range = range(10, 26, 5)      # n of vertices/nodes
    d_range = [0.10, 0.15, 0.20]    # graph density
    Ng = 1000                # n of graphs for each pair n/m

    now = datetime.now()

    jdata = {
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "graphs": []
    }

    graphs = {}
    wl_hashes = set()
    count = 0

    def make_hash(wl_hash):
        wlh = wl_hash
        count = 1
        while wlh in wl_hashes:
            wlh = f"{wl_hash}-{count:02}"
            count += 1

        wl_hashes.add(wlh)
        return wlh

    #
    # enumerate all graphs with order 2,3,4,5
    #
    for n in range(2, 6):
        log.debug(f"... {n} nodes")
        graphs_n = []
        for G in netx.dag_enum(n):
            m = len(G.edges)

            adjacency_matrix = nx.adjacency_matrix(G).toarray().tolist()
            wl_hash = nx.weisfeiler_lehman_graph_hash(G)

            graphs_n.append({
                "n": n,
                "m": m,
                "wl_hash": make_hash(wl_hash),
                "adjacency_matrix": adjacency_matrix
            })

            count += 1
            log.debugt(f'... {count:5}')
        # end
        graphs[str(n)] = graphs_n
        log.info(f"... {n} nodes: {len(graphs_n)} graphs")
    # end

    #
    # Select random DAG with specified order & size
    #
    rnd.seed(42)

    # N: n of nodes
    # D: graph density:
    # G: n of graphs
    # M: n of edges
    for n in n_range:
        log.debug(f"... {n} nodes")
        graphs_n = []

        for d in d_range:
            # select the sizes
            m = iround(d*n*n)

            # generate random DAGs with specified order (n) and size (m)
            for c in range(Ng):
                G = netx.random_dag(n, m)

                adjacency_matrix = nx.adjacency_matrix(G).toarray().tolist()
                wl_hash = nx.weisfeiler_lehman_graph_hash(G)

                if wl_hash in wl_hashes:
                    log.warning(f"... found duplicate wl_hash: {wl_hash}")
                else:
                    wl_hashes.add(wl_hash)

                graphs_n.append({
                    "n": n,
                    "m": m,
                    "wl_hash": wl_hash,
                    "adjacency_matrix": adjacency_matrix
                })

                count += 1
                log.debugt(f'... {count:5}')
            pass
        pass
        graphs[str(n)] = graphs_n
        log.info(f"... {n} nodes: {len(graphs_n)} graphs")
    # end
    jdata['graphs'] = graphs

    log.info(f'saving ...')
    # jsx.save(jdata, f"graphs-{now.strftime('%Y%m%d-%H%M%S')}.json")
    jsx.save(jdata, f"data/graphs-enum.json")
    log.info('done')
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
