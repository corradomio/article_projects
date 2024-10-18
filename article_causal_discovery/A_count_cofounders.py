import stdlib.jsonx as jsx
import networkx as nx
import numpy as np
import netx


def process_graph(ginfo: dict):
    n = ginfo['n'] # n of nodes
    m = ginfo['m'] # n of edges
    wl_hash = ginfo['wl_hash'] # graph id

    print(f"{n}/{m}: {wl_hash}")

    A = np.array(ginfo['adjacency_matrix'], dtype=np.int8)
    G = netx.from_numpy_array(A)
    for u in G.nodes():
        vlist = list(netx.children(G, u))
        if len(vlist) < 2: continue
        print(f"... {u} -> {vlist}")
        nv = len(vlist)
        for i in range(nv-1):
            vi = vlist[i]
            for j in range(i+1, nv):
                vj = vlist[j]
                # check if there exists a path vi->vj or vj->vi
                ij_path = netx.shortest_path(G, vi, vj)
                ji_path = netx.shortest_path(G, vj, vi)
                if len(ij_path) > 0:
                    print(f"... ... {u} -> {ij_path}")
                elif len(ji_path) > 0:
                    print(f"... ... {u} -> {ji_path}")
                else:
                    print(f"... ... {u} -> <{vi}, {vj}> (unconnected)")
            # end
        # end
# end

def process_graph2(ginfo: dict):
    n = ginfo['n'] # n of nodes
    m = ginfo['m'] # n of edges
    wl_hash = ginfo['wl_hash'] # graph id

    print(f"{n}/{m}: {wl_hash}")

    A = np.array(ginfo['adjacency_matrix'], dtype=np.int8)
    G = netx.from_numpy_array(A)
    for u in G.nodes():
        vlist = list(netx.children(G, u))
        if len(vlist) < 2: continue
        print(f"... {u} -> {vlist}")
        nv = len(vlist)
        for i in range(nv-1):
            vi = vlist[i]
            di: set[int] = netx.descendants(G, vi, True)
            for j in range(i+1, nv):
                vj = vlist[j]
                dj: set[int] = netx.descendants(G, vj, True)
                if len(di) == 0 or len(dj) == 0:
                    continue
                print(f"... ... {vi}: {di} -> {vj}: {dj} => {di.intersection(dj)}, {di.difference(dj)}, {dj.difference(di)}")
            # end
        # end
# end



def scan_graphs(file, maxdeg=10):
    graphs = jsx.load(file)
    for order in graphs["graphs"].keys():
        n = int(order)
        if n > maxdeg:
            continue
        graphs_n = graphs["graphs"][order]
        for ginfo in graphs_n:
            # process_graph(ginfo)
            process_graph2(ginfo)
# end


def main():
    scan_graphs("../article_causal_discovery_data/graphs-enum.json", )


if __name__ == "__main__":
    main()
