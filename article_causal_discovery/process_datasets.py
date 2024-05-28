import castle
import h5py
import igraph as ig
import networkx as nx

import stdlib.loggingx as logging
from h5pyx import dump_structure

# HDF5 structure
#
#     /<order>/<graph_id>
#         attrs:
#             'n':                n of nodes (order)
#             'm':                n of edges (size)
#             'adjacency_matrix': adjacency matrix (n x n)
#             'wl_hash':          Weisfeiler Leman hash
#         <method>/<sem_type>:    dataset (10000 x n)
#
# ... Group: /10
# ... ... keys: 3000
# ... ... Group: /10/001a86e80cc18d9bc3bdf374b068c955
# ... ... ... attrs: ['adjacency_matrix', 'm', 'n', 'wl_hash']
# ... ... ... ... adjacency_matrix : numpy.ndarray
# ... ... ... ... m : numpy.int32
# ... ... ... ... n : numpy.int32
# ... ... ... ... wl_hash : str
# ... ... ... keys: 2
# ... ... ... Group: /10/001a86e80cc18d9bc3bdf374b068c955/linear
# ... ... ... ... keys: 5
# ... ... ... ... Dataset: /10/001a86e80cc18d9bc3bdf374b068c955/linear/exp
# ... ... ... ... ... dtype: float32
# ... ... ... ... ... shape: (10000, 10)
#                 ---
# ... ... ... Group: /10/001a86e80cc18d9bc3bdf374b068c955/nonlinear
# ... ... ... ... keys: 3
# ... ... ... ... Dataset: /10/001a86e80cc18d9bc3bdf374b068c955/nonlinear/mim
# ... ... ... ... ... dtype: float32
# ... ... ... ... ... shape: (10000, 10)
#                 ---
#

GRAPH_ORDERS = ['2', '3', '4', '5', '10', '15', '20']


def scan_graphs(c: h5py.Group):
    for order in GRAPH_ORDERS:
        ogroup = c[order]
        for gid in ogroup.keys():
            process_graph(ogroup[gid])


def process_graph(ogroup: h5py.Group):
    pass



def main():
    log = logging.getLogger('main')
    log.info(f"nx: {nx.__version__}")
    log.info(f"ig: {ig.__version__}")
    log.info(f"castle: {castle.__version__}")

    c = h5py.File('graphs-datasets.hdf5', 'r')

    scan_graphs(c)

    log.info("done")
# end



if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
