import numpy as np
import pandasx as pdx
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearnx.feature_selection import SequentialFeatureSelector


def main():
    np.seterr(divide='ignore', invalid='ignore')

    TARGET = 'Obs ID (Recurrence Status)'
    df = pdx.read_data("Dataset-Patent.csv",
                       # dtype=[None, str] + [float]*13210,
                       ignore='Obs ID (Primary)',
                       onehot=TARGET,
                       # header=1
                       )

    X, y = pdx.xy_split(df, target=TARGET)

    feature_names = np.array(X.columns)

    # Normalize all data using a Standard Scaler
    scal = StandardScaler().set_output(transform="pandas")
    Xt = scal.fit_transform(X)
    # Xt = X

    # remove low variance: NOT USEFUL
    # vt = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # Xt = vt.fit_transform(Xt)
    # print("VarianceThreshold", Xt.shape)

    # feature selection
    # knn = SelectKBest(f_classif, k=150).set_output(transform="pandas")
    # Xt = knn.fit_transform(Xt, y[TARGET])
    # print("SelectKBest", Xt.shape, Xt)

    # print("RFECV ...")
    # min_features_to_select = 1
    # clf = LogisticRegression()
    # cv = StratifiedKFold(5)
    # rfecv = RFECV(
    #     estimator=clf,
    #     step=100,
    #     cv=cv,
    #     scoring="accuracy",
    #     min_features_to_select=min_features_to_select,
    #     n_jobs=2,
    #     verbose=1
    # )
    # rfecv.fit(X, y[TARGET])
    # print(f"Optimal number of features: {rfecv.n_features_}")

    estimator = DecisionTreeClassifier()
    scorer = make_scorer(accuracy_score)

    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=20,
        scoring=scorer,
        cv=5,
        verbose=1,
        n_jobs=16
    )
    sfs.fit(Xt, y[TARGET])
    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs.get_support()]}"
    )
    return


if __name__ == "__main__":
    main()
