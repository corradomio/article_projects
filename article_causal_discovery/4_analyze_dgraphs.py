from collections import defaultdict

import networkx as nx
from stdlib.csvx import save_csv
from stdlib.tprint import tprint
from utils import foreach_dataset


#
#   root
#       algo
#           dataset
#               deg
#                   graph
#                       tensor[10,deg,deg]

class CountGraphs:

    def __init__(self):
        self.count = defaultdict(lambda : 0)
        self.graphs: set[str] = set()

    # {
    #     'graph_id': gid,
    #     'n': n,
    #     'm': m,
    #     'adjacency_matrix': adjacency_matrix,
    #     'causal_discovery_algorithm': algorithm,
    #     'data_generation_method': (method, sem_type),
    #     'dataset_index': i,
    #     'causal_adjacency_matrix': causal_dag,
    # }
    def add(self, info):
        gid = info["graph_id"]
        if gid not in self.graphs:
            n = info['n']
            self.graphs.add(gid)
            self.count[n] += 1

    def print(self):
        for d in self.count.keys():
            print(f"{d:2}: {self.count[d]}")


class CountMatches:
    def __init__(self):
        self.n_total = 0
        self.n_processed = defaultdict(lambda : 0)
        self.ground_truth = defaultdict(lambda : 0)
        self.disconnected = defaultdict(lambda : 0)
        self.not_a_dag = defaultdict(lambda : 0)
        self.empty_dag = defaultdict(lambda: 0)

    # {
    #     'graph_id': gid,
    #     'n': n,
    #     'm': m,
    #     'adjacency_matrix': adjacency_matrix,
    #     'causal_discovery_algorithm': algorithm,
    #     'data_generation_method': (method, sem_type),
    #     'dataset_index': i,
    #     'causal_adjacency_matrix': causal_dag,
    # }
    def add(self, path, info):
        n = info['n']
        algorithm = info["causal_discovery_algorithm"]
        adjacency_matrix = info["adjacency_matrix"]
        causal_adjacency_matrix = info["causal_adjacency_matrix"]
        sem_type = path[-2]

        self.n_total += 1
        self.n_processed[(algorithm, n, sem_type)] += 1

        if abs(adjacency_matrix - causal_adjacency_matrix).sum() == 0:
            self.ground_truth[(algorithm, n, sem_type)] += 1
        elif causal_adjacency_matrix.sum() == 0:
            self.empty_dag[(algorithm, n, sem_type)] += 1
            return

        G = nx.from_numpy_array(causal_adjacency_matrix, create_using=nx.DiGraph())
        if not nx.is_weakly_connected(G):
            self.disconnected[(algorithm, n, sem_type)] += 1
        elif not nx.is_directed_acyclic_graph(G):
            self.not_a_dag[(algorithm, n, sem_type)] += 1
        else:
            pass
        return

    def table(self, fname):
        header = ['algo', 'degree', 'sem_type']
        table_ = defaultdict(lambda: [])
        total_ = defaultdict(lambda: 0)

        header.append("processed")
        for a_d_st in self.n_processed:
            algo, deg, sem_type = a_d_st
            n_processed = self.n_processed[a_d_st]
            table_[a_d_st].append(n_processed)
            total_["processed"] += n_processed

        header.append("ground_truth")
        for a_d_st in self.n_processed:
            algo, deg, sem_type = a_d_st
            n_processed = self.ground_truth[a_d_st]
            table_[a_d_st].append(n_processed)
            # total_["ground_truth"] += n_processed

        header.append("disconnected")
        for a_d_st in self.n_processed:
            algo, deg, sem_type = a_d_st
            n_processed = self.disconnected[a_d_st]
            table_[a_d_st].append(n_processed)
            # total_["disconnected"] += n_processed

        header.append("not_a_dag")
        for a_d_st in self.n_processed:
            algo, deg, sem_type = a_d_st
            n_processed = self.not_a_dag[a_d_st]
            table_[a_d_st].append(n_processed)
            # total_["not_a_dag"] += n_processed

        header.append("empty_dag")
        for a_d_st in self.n_processed:
            algo, deg, sem_type = a_d_st
            n_processed = self.empty_dag[a_d_st]
            table_[a_d_st].append(n_processed)
            # total_["empty_dag"] += n_processed

        # table_[('total', 0, 'any')].extend([
        #     total_['processed'],
        #     total_['ground_truth'],
        #     total_['disconnected'],
        #     total_['not_a_dag'],
        #     total_['empty_dag'],
        # ])

        tlist = [
            ([k[0], k[1], k[2]] + table_[k])
            for k in table_.keys()
        ]

        save_csv(fname, tlist, header=header)

    def print(self):
        print(f"processed {self.n_total} graphs")
        for degree in self.n_processed:
            print(f"    {degree}: {self.n_processed[degree]}")
        print(f"found ground truth graphs")
        for degree in self.ground_truth:
            print(f"    {degree}: {self.ground_truth[degree]}")
        print(f"found disconnected graphs")
        for degree in self.disconnected:
            print(f"    {degree}: {self.disconnected[degree]}")
        print(f"found not dag graphs")
        for degree in self.not_a_dag:
            print(f"    {degree}: {self.not_a_dag[degree]}")
        print(f"found empty graphs")
        for degree in self.empty_dag:
            print(f"    {degree}: {self.empty_dag[degree]}")
        print("end")

# end


def main():
    tprint("Starting analysis ...", force=True)

    # foreach_datasets("./data")

    # cg = CountGraphs()
    # foreach_datasets("./data", callback=lambda path, info: cg.add(info))
    # cg.print()

    cm = CountMatches()
    foreach_dataset(
        "./data",
        callback=lambda path, info: cm.add(path, info),
        max_degree=3,
        skip_algos=['DirectLiNGAM', 'GES', 'GOLEM', 'ICALiNGAM']
    )
    cm.table("graphs_statistics.csv")

    tprint("Done", force=True)
    pass


if __name__ == "__main__":
    # logging.config.fileConfig("logging_config.ini")
    # logging.getLogger("main").info("Logging initialized")
    main()
