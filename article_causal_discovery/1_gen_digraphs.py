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


def flip():
    return rnd.random() > 0.5


def main():
    log = logging.getLogger('main')
    log.info(f"nx: {nx.__version__}")
    log.info(f"castle: {castle.__version__}")

    os.makedirs('data', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)

    all_range = range(0,0)              # ALL
    n_range = range(7,16)               # sampled
    d_range = [.10, .15, .20, .25, .30] # graph density
    Ng = 10000                       # n of graphs for each pair n/m
    rnd.seed(42)

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
    # enumerate all graphs with order 2,3,4,5 [,6]
    #
    for n in all_range:
        log.debug(f"... {n} nodes")

        graphs = {}
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

        log.info(f'saving ...')
        jdata['graphs'] = graphs
        jsx.save(jdata, f"data/graphs-enum-{n}.json")
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
        # log.debug(f"... {n} nodes")

        graphs = {}
        graphs_n = []

        for d in d_range:
            log.debug(f"... {n} nodes, {d} density")

            # select the sizes
            m = iround(d*n*n)

            graphs_d = []
            # generate random DAGs with specified order (n) and size (m)
            while len(graphs_d) < Ng:
                G = netx.random_dag(n, m)

                adjacency_matrix = nx.adjacency_matrix(G).toarray().tolist()
                wl_hash = nx.weisfeiler_lehman_graph_hash(G)

                if wl_hash in wl_hashes:
                    if n > 6:
                        continue
                    if flip():
                        continue
                    wl_hash = make_hash(wl_hash)
                # end

                wl_hashes.add(wl_hash)

                size = G.number_of_edges()

                graphs_d.append({
                    "n": n,
                    "m": size,
                    "wl_hash": wl_hash,
                    "adjacency_matrix": adjacency_matrix
                })

                count += 1
                log.debugt(f'... {count:5}')
            pass
            graphs_n.extend(graphs_d)

            # graphs_n = graphs_d
            #
            # graphs[str(n)] = graphs_n
            # jdata['graphs'] = graphs
            # jsx.save(jdata, f"data/graphs-enum-{n}-{int(d*100)}.json")
        pass

        graphs[str(n)] = graphs_n
        log.info(f"... {n} nodes: {len(graphs_n)} graphs")

        graphs[str(n)] = graphs_n
        jdata['graphs'] = graphs
        jsx.save(jdata, f"data/graphs-enum-{n}-sampled.json")
    # end
    # jdata['graphs'] = graphs
    # log.info(f'saving ...')
    # jsx.save(jdata, f"graphs-{now.strftime('%Y%m%d-%H%M%S')}.json")
    # jsx.save(jdata, f"data/graphs-enum-6-9.json")

    log.info('done')
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
