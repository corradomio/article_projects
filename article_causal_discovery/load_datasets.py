from datetime import datetime
from path import Path as path
import os
import castle
import igraph as ig
import networkx as nx
import numpy as np
import stdlib.jsonx as jsx
import stdlib.loggingx as logging
import h5py
from causalx.iidsim import IIDSimulation
from h5pyx import dump_structure


def main():
    log = logging.getLogger('main')
    log.info(f"nx: {nx.__version__}")
    log.info(f"ig: {ig.__version__}")
    log.info(f"castle: {castle.__version__}")

    container = h5py.File('graphs-datasets.hdf5', 'r')
    dump_structure(container)


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
