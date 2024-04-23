import os

import numpy as np
import matplotlib.pyplot as plt
import pandasx as pdx


def main():
    np.seterr(divide='ignore', invalid='ignore')

    TARGET = 'Obs ID (Recurrence Status)'
    df = pdx.read_data("Dataset-Patent.csv",
                       ignore='Obs ID (Primary)',
                       onehot=TARGET,
                       )

    X, y = pdx.xy_split(df, target=TARGET)
    y = y[TARGET]
    n = len(X.columns)

    for i in range(n):
        plt.clf()
        col = X.columns[i]
        print(col)

        Xc = X[col]
        X0 = Xc[y == 0]
        X1 = Xc[y == 1]

        plt.hist(X0, alpha=0.75)
        plt.hist(X1, alpha=0.75)
        plt.title(col)
        plt.tight_layout()

        name = col.replace(' /// ', '-')
        dir = f"plots/{name[0:2]}"
        os.makedirs(dir, exist_ok=True)
        plt.savefig(f"{dir}/{name}.png", dpi=300)
        pass

    return


if __name__ == "__main__":
    main()
