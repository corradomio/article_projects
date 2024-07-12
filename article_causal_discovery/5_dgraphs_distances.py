import stdlib.loggingx as logging
import os
import h5py
import numpy as np
import netx as nxx

from stdlib.tprint import tprint
from utils import foreach_dataset, ALGORITHMS, SEM_TYPES

#
#
#   degree
#       DAG ground-truth [degree]
#           algorithm (6)
#               sem_type (7)
#                   predicted causal graph (10)
#
#   degree
#       DAG ground-truth [degree]
#           sem_type (7)
#               predicted causal graph (10)
#                   hamming([GT, algo1, ... algo6])
#

# (degree, gid, cdalgo, dgmethod, i),
# {
#     'graph_id': gid,
#     'n': n,
#     'm': m,
#     'adjacency_matrix': adjacency_matrix,
#     'causal_discovery_algorithm': algorithm,
#     'algorithm_index': ALGORITHMS[algorithm],
#     'data_generation_method': (method, sem_type),
#     'method': method,
#     'sem_type': sem_type,
#     'dataset_index': i,
#     'causal_adjacency_matrix': causal_dag,
# }

n_sem_types = len(SEM_TYPES)
n_algorithms = len(ALGORITHMS)
n_datasets = 10


def hamming_distance(am: np.ndarray, cm: np.ndarray) -> float:
    return abs(am - cm).sum()


class HammingDistance:

    def __init__(self):
        self.adj_mat = {}       # adjacency matrix
        self.dsepmat = {}       # using 'd_separation()'
        self.dsep_uv = {}       # using 'd_separation_pair()'
        self.adj_pow = {}       # power of (I + adjacency matrix)

        self.adjm_dist = {}     # using adjacency matrix
        self.dsep_dist = {}     # using d_separation matrix
        self.dsuv_dist = {}     # using d_separation_pair matrix
        self.adjp_dist = {}     # using power of (I + adjacency matrix)
        self.count = 0

        self.log = logging.getLogger("hdist__")

    def add(self, path, info):
        # compose the adjacency matrices for GT and discovered dags
        # in a tensor having the structure:
        #
        #   (sem_types, datasets, algorithms+1, n, n)
        #
        # where:
        #   sem_types:   7 data distributions   (exp, gauss, gumel, uniform | mim, mlp, quadratic)
        #   datasets:   10 causal graphs generated from different datasets
        #   algorithms:  GT, PC, DirectLiNGAM, ICALiNGAM, GES, GOLEM, Notears
        #   n, n:       adjacency matrix for GT (index 0) and the other algorithms
        #               (previous index)
        #
        self.count += 1

        gid = info['graph_id']
        n = info['n']
        am = info['adjacency_matrix']
        cm = info['causal_adjacency_matrix']
        si = info['sem_index']
        ai = info['algorithm_index']
        di = info['dataset_index']

        # for each graph, the tensor composed by
        #   n_sem_types
        #   n_data_sets
        #   n_algorithms+1 because the 'algorithm' 0 is the 'identity'
        #       that is, it contains the 'ground truth'
        if gid not in self.adj_mat:
            self.adj_mat[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n, n), dtype=np.int8)
            self.dsepmat[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n, n), dtype=np.int8)
            self.dsep_uv[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n, n), dtype=np.int8)
            self.adj_pow[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n, n), dtype=np.int8)

            dsep = nxx.d_separation(am)
            dsuv = nxx.d_separation_pairs(am)
            adj1 = nxx.power_adjacency_matrix(am)
            for gsi in range(n_sem_types):
                for gdi in range(n_datasets):
                    self.adj_mat[gid][gsi, gdi, 0] = am
                    self.dsepmat[gid][gsi, gdi, 0] = dsep
                    self.dsep_uv[gid][gsi, gdi, 0] = dsuv
                    self.adj_pow[gid][gsi, gdi, 0] = adj1
            # end gsi/gdi
        self.adj_mat[gid][si, di, ai + 1] = cm
        self.dsepmat[gid][si, di, ai + 1] = nxx.d_separation(cm)
        self.dsep_uv[gid][si, di, ai + 1] = nxx.d_separation_pairs(cm)
        self.adj_pow[gid][si, di, ai + 1] = nxx.power_adjacency_matrix(cm)

        self.log.infot(f"... ... {len(self.adj_mat)}/{self.count}")
        pass

    def distances(self):
        self.log.info("DiGraph distances ...")
        count = 0
        for gid in self.adj_mat:
            self.log.info(f"... {gid}")

            self.adjm_dist[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n_algorithms + 1), dtype=int)
            self.dsep_dist[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n_algorithms + 1), dtype=int)
            self.dsuv_dist[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n_algorithms + 1), dtype=int)
            self.adjp_dist[gid] = np.zeros((n_sem_types, n_datasets, n_algorithms + 1, n_algorithms + 1), dtype=int)

            for si in range(n_sem_types):
                for di in range(n_datasets):
                    for ai in range(n_algorithms):
                        for aj in range(ai+1, n_algorithms+1):
                            # -------------------------------------
                            # adjacency matrix hamming distance
                            # -------------------------------------
                            mi = self.adj_mat[gid][si, di, ai]
                            mj = self.adj_mat[gid][si, di, aj]

                            hd = hamming_distance(mi, mj)

                            # the distance is 'symmetric'
                            self.adjm_dist[gid][si, di, ai, aj] = hd
                            self.adjm_dist[gid][si, di, aj, ai] = hd

                            # -------------------------------------
                            # d_separation matrix hamming distance
                            # -------------------------------------
                            mi = self.dsepmat[gid][si, di, ai]
                            mj = self.dsepmat[gid][si, di, aj]

                            hd = hamming_distance(mi, mj)

                            # the distance is 'symmetric'
                            self.dsep_dist[gid][si, di, ai, aj] = hd
                            self.dsep_dist[gid][si, di, aj, ai] = hd

                            # -------------------------------------
                            # d_separation_pair matrix hamming distance
                            # -------------------------------------
                            mi = self.dsep_uv[gid][si, di, ai]
                            mj = self.dsep_uv[gid][si, di, aj]

                            hd = hamming_distance(mi, mj)

                            # the distance is 'symmetric'
                            self.dsuv_dist[gid][si, di, ai, aj] = hd
                            self.dsuv_dist[gid][si, di, aj, ai] = hd

                            # -------------------------------------
                            # (I+A)^(n-1))
                            # -------------------------------------
                            mi = self.adj_pow[gid][si, di, ai]
                            mj = self.adj_pow[gid][si, di, aj]

                            hd = hamming_distance(mi, mj)

                            # the distance is 'symmetric'
                            self.adjp_dist[gid][si, di, ai, aj] = hd
                            self.adjp_dist[gid][si, di, aj, ai] = hd

                            # -------------------------------------
                            # end
                            # -------------------------------------
                            count += 1
                            self.log.infot(f"... {self.count}")
        # end gid/si/di/ai/aj
        self.log.info(f"Done {self.count}))", force=True)
        return
    # end

    def save(self, hdpath):
        if os.path.exists(hdpath):
            os.remove(hdpath)
        dest = h5py.File(hdpath, 'w')
        dest.attrs['sem_type'] = list(SEM_TYPES.keys())
        dest.attrs['algorithm'] = list(ALGORITHMS.keys())

        dest.attrs['n_sem_types'] = n_sem_types
        dest.attrs['n_algorithms'] = n_algorithms
        dest.attrs['n_datasets'] = n_datasets

        for gid in self.adj_mat:
            n = self.adj_mat[gid].shape[-1]
            dest[f'{n}/{gid}/matrix/adjacency_matrix'] = self.adj_mat[gid]
            dest[f'{n}/{gid}/matrix/d_separation'] = self.dsepmat[gid]
            dest[f'{n}/{gid}/matrix/d_separation_pair'] = self.dsep_uv[gid]
            dest[f'{n}/{gid}/matrix/power_1'] = self.adj_pow[gid]

            dest[f'{n}/{gid}/dist/adjacency_matrix'] = self.adjm_dist[gid]
            dest[f'{n}/{gid}/dist/d_separation'] = self.dsep_dist[gid]
            dest[f'{n}/{gid}/dist/d_separation_pair'] = self.dsuv_dist[gid]
            dest[f'{n}/{gid}/dist/power_1'] = self.adjp_dist[gid]
        # end
        dest.close()
    # end


def main():
    hd = HammingDistance()

    foreach_dataset(
        "../data",
        callback=lambda path, info: hd.add(path, info),
        # max_degree=5,
        # skip_algos=['DirectLiNGAM', 'GES', 'GOLEM', 'ICALiNGAM']
    )
    hd.distances()
    hd.save("../data/graph-distances.hdf5")
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
