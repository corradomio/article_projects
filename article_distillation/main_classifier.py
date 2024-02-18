#
# check the classified on datasets with different dimensions
#
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from path import Path as path
import numpy as np
import pandas as pd
import pandasx as pdx

#
# Directory containing dataset with 100,1000,10000 data points
# on 2,3,4,5,10,25,50,100  dimensions
#
# The values in each dimension is in range [0,1]
#
# The space is subdivided in a regular grid of k subdivisions for each dimension
# For now, k=3
#
DATA_DIR = "data_reg"

#
# Directory containing dataset with 100,1000,10000 data points
# on 2,3,4,5,10,25,50,100  dimensions
#
# The values in each dimension is in range [0,1]
#
# The space is subdivided in a irregular grid of k subdivisions with
# random subdivisions in each dimension.
# The maximum number of subdivisions is 5
#
# DATA_DIR = "data_rand"


def load_data(f, noise=0):
    # print(f"Loading {f.stem} ...")
    df: pd.DataFrame = pdx.read_data(f)

    y = df[['y']].to_numpy(dtype=int)
    X = df[df.columns.difference(['y'])].to_numpy()

    # add artificial noise
    if noise > 0:
        N = np.random.normal(loc=0, scale=noise, size=X.shape)
        X += N

    # n of dimensions of the data points
    ndims = len(df.columns)-1
    return X, y, ndims
# end


def classify(X, y):
    print("... evaluate the classifier")
    ntrain = int(len(X)*.8)

    X_train, X_test = X[:ntrain], X[ntrain:]
    y_train, y_test = y[:ntrain], y[ntrain:]

    # note: the default configuration doesn't limit the depth!
    dt = DecisionTreeClassifier(
        criterion='gini'
        # criterion='entropy'       # bad
        # criterion='log_loss'      # bad
    )

    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    print(f"... ... accuracy: {score:.3}")
# end


def main():
    data_dir = path(DATA_DIR)
    for f in data_dir.files("*.csv"):
        X, y, ndims = load_data(f, 0)

        #
        # Note: the accuracy with ndims > 10 is 50% that is, totally random!
        #       This is true for 'data_reg' and 'data_rand'
        #
        if ndims <= 10: continue

        classify(X, y)
    pass



if __name__ == "__main__":
    main()
