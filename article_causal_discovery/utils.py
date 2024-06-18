import h5py
from path import Path as path
from stdlib.tprint import tprint
from joblib import Parallel, delayed


def null_callback(path, info): pass


def foreach_hdf(
    dir,
    callback,
    max_degree,
    skip_algos,
    skip_methods
):
    if dir.stem in skip_methods:
        return

    tprint("...", dir.stem, force=True)

    # scan the files
    for hdf in dir.files("*.hdf5"):
        data = h5py.File(hdf, "r")
        # scan the graph degrees

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

                tprint("... ... ...", gid, ":", (n, m))

                # scan the causal discovery algorithms
                for cdalgo in ginfo.keys():
                    # tprint("... ... ... ...", cdalgo)
                    cdinfo = ginfo[cdalgo]

                    # scan the data generation methods
                    for dgmethod in cdinfo.keys():
                        causal_dags = cdinfo[dgmethod]
                        algorithm = causal_dags.attrs['algorithm']
                        method = causal_dags.attrs['method']
                        semtype = causal_dags.attrs['sem_type']
                        data_shape = causal_dags.shape

                        if algorithm in skip_algos: continue
                        if method in skip_methods: continue

                        # tprint("... ... ... ... ...", method, semtype, force=True)

                        for i in range(data_shape[0]):
                            causal_dag = causal_dags[i]

                            callback((graph_degree, gid, cdalgo, dgmethod, i),
                                     {
                                         'graph_id': gid,
                                         'n': n,
                                         'm': m,
                                         'adjacency_matrix': adjacency_matrix,
                                         'causal_discovery_algorithm': algorithm,
                                         'data_generation_method': (method, semtype),
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
    tprint("Sequential start processing ... ", force=True)

    # scan the dirs
    for dir in path(root).dirs():
        if dir.stem not in skip_algos:
            foreach_hdf(dir, callback, max_degree, skip_algos, skip_methods)

    tprint("Done", force=True)
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


