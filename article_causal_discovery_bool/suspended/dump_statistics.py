from path import Path as path
import h5py

ROOT = path(r"../article_causal_discovery_bool_data/datasets_suspended")

def keysof(file: path) -> list[str]:
    f = h5py.File(file)
    keys = []
    for order in f.keys():
        graphs = f[order]
        keys = [
            f"/{order}/{k}"
            for k in graphs.keys()
        ]
    f.close()
    return keys
# end


def main():
    for file in ROOT.files("*.hdf5"):
        print(f"{file.stem}: {len(keysof(file))}")
    pass


if __name__ == "__main__":
    main()
