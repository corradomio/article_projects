import stdlib.loggingx as logging
import h5py
from path import Path as path
from stdlib.tprint import tprint
from joblib import Parallel, delayed


def null_callback(path, info): pass


ALGORITHMS = {
    'PC': 0,
    'DirectLiNGAM': 1,
    'ICALiNGAM': 2,
    'GES': 3,
    'GOLEM': 4,
    'Notears': 5,
}

# selected: PC, DirectLiNGAM, ICALiNGAM, GES, GOLEM, Notears


# linear:       gauss, exp, gumbel, uniform, logistic
# nonlinear:    mlp, mim, gp, gp-add, quadratic
# selected:     exp, gauss, gumel, uniform,mim, mlp, quadratic


SEM_TYPES = {
    'exp': 0,           # linear
    'gauss': 1,         # linear
    'gumbel': 2,        # linear
    'uniform': 3,       # linear
    'mim': 4,           # nonlinear
    'mlp': 5,           # nonlinear
    'quadratic': 6,     # nonlinear
}


def foreach_hdf(
    dir,
    callback,
    max_degree,
    skip_algos,
    skip_methods,
    log
):
    if dir.stem in skip_methods:
        return

    log.info(f"... {dir.stem}")

    # scan the files
    for hdf in dir.files("*.hdf5"):
        data = h5py.File(hdf, "r")
        # scan the graph degrees
        # tprint(f"... ... {hdf.stem}", force=True)

        for sdeg in data:
            graph_degree = int(sdeg)
            if graph_degree > max_degree: continue
            # tprint("... ... ...", graph_degree)

            ginfos = data[sdeg]
            # scan the graphs
            for gid in ginfos.keys():
                ginfo = ginfos[gid]
                n = ginfo.attrs['n']
                m = ginfo.attrs['m']
                adjacency_matrix = ginfo.attrs['adjacency_matrix']

                assert n == graph_degree

                log.infot(f"... ... {gid}: { (n, m)}")

                # scan the causal discovery algorithms
                for cdalgo in ginfo.keys():
                    # tprint("... ... ... ...", cdalgo)
                    cdinfo = ginfo[cdalgo]

                    # scan the data generation methods
                    for dgmethod in cdinfo.keys():
                        causal_dags = cdinfo[dgmethod]
                        algorithm = causal_dags.attrs['algorithm']
                        method = causal_dags.attrs['method']
                        sem_type = causal_dags.attrs['sem_type']
                        data_shape = causal_dags.shape

                        if algorithm in skip_algos: continue
                        if method in skip_methods: continue

                        # tprint("... ... ... ... ...", method, sem_type, force=True)

                        for i in range(data_shape[0]):
                            causal_dag = causal_dags[i]

                            callback((graph_degree, gid, cdalgo, dgmethod, i),
                                     {
                                         'graph_id': gid,
                                         'n': n,
                                         'm': m,
                                         'adjacency_matrix': adjacency_matrix,
                                         'causal_discovery_algorithm': algorithm,
                                         'data_generation_method': (method, sem_type),
                                         'method': method,
                                         'sem_type': sem_type,

                                         'algorithm_index': ALGORITHMS[algorithm],
                                         'sem_index': SEM_TYPES[sem_type],
                                         'dataset_index': i,
                                         'causal_adjacency_matrix': causal_dag,
                                     })
                        # end data
                    # end dmethod
                # end cdalgo
            # end gid
        # end sdeg
    # end hdf
# end


def foreach_dataset(
    root, *,
    callback=null_callback,
    max_degree=10,
    skip_algos=(),
    skip_methods=()
):
    log = logging.getLogger('foreach')
    log.info("Sequential start processing ... ")

    # scan the dirs
    for dir in path(root).dirs():
        if dir.stem not in skip_algos:
            foreach_hdf(dir, callback, max_degree, skip_algos, skip_methods, log)

    log.info("Done")
# end


def parallel_foreach_dataset(
    root, *,
    callback=null_callback,
    max_degree=10,
    skip_algos=(),
    skip_methods=(),
    n_jobs=0):

    tprint("Parallel start processing ... ", force=True)

    Parallel(n_jobs=n_jobs)(
        delayed(foreach_hdf)(dir, callback, max_degree, skip_algos, skip_methods) \
        for dir in path(root).dirs()
    )

    tprint("Done", force=True)
# end


