import logging.config
import os

import h5py

from collections import defaultdict
from utils import foreach_dataset, ALGORITHMS, SEM_TYPES

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
#
#
# (degree, gid, cdalgo, dgmethod, i),
# {
#     'graph_id': gid,
#     'n': n,
#     'm': m,
#     'adjacency_matrix': adjacency_matrix,
#     'causal_discovery_algorithm': algorithm,
#     'data_generation_method': (method, sem_type),
#     'method': method,
#     'sem_type': sem_type,
#
#     'algorithm_index': ALGORITHMS[algorithm],
#     'sem_index': SEM_TYPES[sem_type],
#     'dataset_index': i,
#     'causal_adjacency_matrix': causal_dag,
# }


n_sem_types = len(SEM_TYPES)
n_algorithms = len(ALGORITHMS)
n_datasets = 10


class PartialDags:

    def __init__(self):
        self.count = 0
        self.pdags = defaultdict(lambda: 0)
        self.ndags = defaultdict(lambda: 0)

    def add(self, path, info):
        self.count += 1

        n = info['n']
        cda = info['causal_discovery_algorithm']
        cm = info['causal_adjacency_matrix']
        st = info['sem_type']

        if len(cm[cm < 0]) > 0:
            self.pdags[(cda, st, n)] += 1
        elif len(cm[cm > 0]) > 0:
            self.ndags[(cda, st, n)] += 1
        pass

    def dump(self):
        print("Partial dags")
        for k in self.pdags:
            print(f"{k}: {self.pdags[k]}")

        print("Normal dags")
        for k in self.ndags:
            print(f"{k}: {self.ndags[k]}")
    # end


def main():
    pdags = PartialDags()

    foreach_dataset(
        "./data",
        callback=lambda path, info: pdags.add(path, info),
        # max_degree=5,
        # skip_algos=['DirectLiNGAM', 'GES', 'GOLEM', 'ICALiNGAM']
    )
    pdags.dump()
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
