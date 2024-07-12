import h5py
from path import Path as path
from stdlib import jsonx
import numpy as np

# HDF5 structure:
#   item_country
#       true:           y_true
#       library
#           algo:       y_pred
#               wape    float
#


def main():

    best_models = {}

    for f in path(".").files("*.hdf5"):
        print("file:", f)
        with h5py.File(f) as f:
            for item_country in f:
                print("...", item_country)
                if item_country not in best_models:
                    y_true: np.ndarray = f[f"{item_country}/true"][:]
                    best_models[item_country] = dict(
                        y_true=y_true,
                        y_pred=[],
                        algo='_:_',
                        wape=10,
                        worst=[]
                    )
                # end
                group_item_country = f[item_country]

                for lib in group_item_country:
                    if lib == 'true': continue
                    group_lib = group_item_country[lib]
                    for algo in group_lib:

                        g: h5py.Dataset = group_lib[algo]
                        wape = g.attrs['wape']
                        print(f"... ... {lib}/{algo}: {wape}")
                        if wape < best_models[item_country]['wape']:
                            # print(f"... ... {lib}/{algo}: {wape}")
                            calgo = best_models[item_country]['algo']
                            cwape = best_models[item_country]['wape']
                            best_models[item_country]['worst'].append((calgo, cwape))

                            best_models[item_country]['algo'] = f"{lib}:{algo}"
                            best_models[item_country]['wape'] = wape
                            best_models[item_country]['y_pred'] = g[:]
                        pass
                    pass
                pass
    # end with/for
    jsonx.save(best_models, "best_models.json")
    print("done")
    pass


if __name__ == "__main__":
    main()

