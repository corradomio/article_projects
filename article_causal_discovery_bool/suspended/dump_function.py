from path import Path as path
import h5py
from stdlib import picklex
from stdlib import jsonx

ROOT = path(r"../article_causal_discovery_bool_data/datasets")


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
    for pfile in ROOT.files("*.pickle"):
        print(f"{pfile.stem}:")
        jdata = picklex.load(pfile)

        jfile = pfile.parent / f"{pfile.stem}.json"
        jsonx.dump(jdata, jfile)
    pass


if __name__ == "__main__":
    main()
