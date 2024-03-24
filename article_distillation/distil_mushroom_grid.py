import numpy as np
from path import Path as path
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from common import *
from pandasx.preprocessing import BinHotEncoder
from stdlib.timing import tprint


# from category_encoders import OneHotEncoder


def load_data():
    df = pdx.read_data("data_uci/mushroom/mushroom.csv",
                       categorical=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                                    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                                    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
                                    ]
                       )
    # df = pdx.read_data("data_uci/mushroom/mushroom.csv",
    #                    binhot=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    #                            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
    #                            'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    #                            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    #                            'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    #                            ])

    X, y = pdx.xy_split(df, target='target')
    # X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=.25)
    # return X_train, X_test, y_train, y_test
    return X, y


class TargetFunction(BaseTargetFunction):

    def __init__(self, X, y, D, **params):
        super().__init__(X, y, D)

        X = self.X
        y = self.y

        self.xenc = BinHotEncoder().fit(X)
        self.yenc = BinHotEncoder().fit(y)

        self.X_true = self.xenc.transform(X)
        self.y_true = self.yenc.transform(y)

        # create the Ground Truth classifier
        self.GTC = self.create_classifier(X, y)

        self._params = params
        self._distilled_model = None
        self._Xd = None
        self._yd = None
    # end

    # -----------------------------------------------------------------------
    # For 'gp_minimize'

    def create_classifier(self, X, y):
        Xenc = self.xenc.transform(X)
        yenc = self.yenc.transform(y)

        c = DecisionTreeClassifier()
        c.fit(Xenc, yenc)
        return c

    def create_labels(self, X):
        Xenc = self.xenc.transform(X)
        yenc = self.GTC.predict(Xenc)
        yenc = self.make_target(yenc)
        y = self.yenc.inverse_transform(yenc)
        return y

    def make_target(self, y):
        if isinstance(y, np.ndarray):
            columns = self.y.columns
            y = pd.DataFrame(data=y.reshape(-1, len(columns)), columns=columns)
        return y

    # def __call__(self, *args, **kwargs):
    #     # distilled points
    #     Xd = reshape(*args, self.X.columns)
    #
    #     if self.parameters is None:
    #         yd = self.create_labels(Xd)
    #     else:
    #         Xd, yd = self.nearest_neighbors(Xd)
    #
    #     model = self.create_classifier(Xd, yd)
    #
    #     y_true = self.y_true
    #     y_pred = model.predict(self.X_true)
    #
    #     score = accuracy_score(y_true, y_pred)
    #     self.score_history.append(score)
    #     iter = len(self.score_history)
    #
    #     if score > self.best_score:
    #         self.best_score = score
    #         self.best_model = model
    #         self.best_params = (Xd, yd)
    #         self.best_iter = iter
    #         tprint(f"[{iter:2}] Best score: {score}")
    #         self.best_score_history.append({"iter": iter, "score": score})
    #     else:
    #         tprint(f"[{iter:2}] .... score: {score}")
    #
    #     return score if self.maximize else (1-score)

    def __call__(self, *args, **kwargs):
        # distilled points
        Xd = reshape(*args, self.X.columns)
        yd = self.create_labels(Xd)

        self._Xd = Xd
        self._yd = yd

        self._distilled_model = self.create_classifier(Xd, yd)

        return self._update_score()

    def _update_score(self):

        y_true = self.y_true
        model = self._distilled_model
        Xd = self._Xd
        yd = self._yd
        y_pred = model.predict(self.X_true)

        score = accuracy_score(y_true, y_pred)
        self.score_history.append(score)
        iter = len(self.score_history)

        if score > self.best_score:
            self.best_score = score
            self.best_model = model
            self.best_params = (Xd, yd)
            self.best_iter = iter
            tprint(f"[{iter:2}] Best score: {score}")
            self.best_score_history.append({"iter": iter, "score": score})
        else:
            tprint(f"[{iter:2}] .... score: {score}")

        return score
    # end

    # def nearest_neighbors(self, Xd) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     Xns = []
    #     yns = []
    #     for i in range(self.D):
    #         xd = Xd.iloc[i]
    #         xn, yn = self.parameters.xn(xd)
    #         Xns.append(xn)
    #         yns.append(yn)
    #     Xns = pd.DataFrame(data=Xns).reset_index(drop=True)
    #     yns = pd.DataFrame(data=yns).reset_index(drop=True)
    #     return Xns, yns

    # -----------------------------------------------------------------------
    # As scikit-learn estimator

    def fit(self, X, y):
        super().fit(X, y)

        self.xenc.fit(X)
        self.yenc.fit(y)

        self.X_true = self.xenc.transform(X)
        self.y_true = self.yenc.transform(y)

        # create the Ground Truth classifier
        self.GTC = self.create_classifier(X, y)

        # scikit-learn

        # self._params = self.parameters.grid_values()
        # self._params = params
        self._distilled_model = None
        self._Xd = None
        self._yd = None

        params = list(self._params.values())
        Xd = reshape(params, self.X.columns)
        yd = self.create_labels(Xd)

        self._distilled_model = self.create_classifier(Xd, yd)
        self._Xd = Xd
        self._yd = yd
        return self

    # -----------------------------------------------------------------------

    def save(self, fname):
        super().save(fname)

    def plot(self, fname):
        super().plot(fname)
# end


def main():
    path("results/grid").mkdir_p()
    path("plots/grid").mkdir_p()

    X, y = load_data()

    D = 100
    for D in [10, 25, 50, 100]:
        # prepare the data structures
        parameters = Parameters(X, y, D)
        target_function = TargetFunction(X, y, D, **parameters.grid_values())
        target_function.fit(X, y)

        N_TRAY = 100
        for i in range(N_TRAY):
            x0 = parameters.x0()
            target_function(x0)

        target_function.save(f"results/grid/mushroom-distilled")
        target_function.plot(f"plots/grid/mushroom-distilled")
    # end

    pass
# end


if __name__ == '__main__':
    main()
