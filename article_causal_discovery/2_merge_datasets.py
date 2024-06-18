import logging.config
from path import Path as path
import os
import h5py
from stdlib.tprint import tprint



def main():

    dir = path("data")
    gdspath = dir.joinpath("graphs-datasets.hdf5")

    if os.path.exists(gdspath):
        os.remove(gdspath)
    dest = h5py.File(gdspath, 'w')

    tprint("Staring copy ...", force=True)
    count = 0
    for gin in dir.files("graphs-datasets-*.hdf5"):
        source = h5py.File(gin, 'r')
        for d in source.keys():
            graphs_d = source[d]
            for gid in graphs_d.keys():
                ginfo = graphs_d[gid]

                dinfo = dest.create_group(ginfo.name)
                dinfo.attrs['n'] = ginfo.attrs['n']
                dinfo.attrs['m'] = ginfo.attrs['m']
                dinfo.attrs['wl_hash'] = ginfo.attrs['wl_hash']
                dinfo.attrs['adjacency_matrix'] = ginfo.attrs['adjacency_matrix']

                for method in ginfo.keys():
                    sds = ginfo[method]
                    dds = dinfo.create_dataset(sds.name, shape=sds.shape, data=sds)
                    dds.attrs['method'] = sds.attrs['method']
                    dds.attrs['sem_type'] = sds.attrs['sem_type']

                    count += 1
                    tprint(f"... {count}")
                    # break
                # break
            # break
        # break
    # end
    dest.close()
    tprint(f"done {count}", force=True)
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
