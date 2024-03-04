import numpy as np
from path import Path as path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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

    def __init__(self, data, D, parameters=None, maximize=True):
        super().__init__(data, D, parameters=parameters, maximize=maximize)

        X = self.X
        y = self.y

        self.xenc = BinHotEncoder().fit(X)
        self.yenc = BinHotEncoder().fit(y)

        self.X_true = self.xenc.transform(X)
        self.y_true = self.yenc.transform(y)

        # create the Ground Truth classifier
        self.GTC = self.create_classifier(X, y)
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

    def __call__(self, *args, **kwargs):
        # distilled points
        Xd = reshape(*args, self.X.columns)

        if self.parameters is None:
            yd = self.create_labels(Xd)
        else:
            Xd, yd = self.nearest_neighbors(Xd)

        model = self.create_classifier(Xd, yd)

        y_true = self.y_true
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

        return score if self.maximize else (1-score)

    def nearest_neighbors(self, Xd) -> tuple[pd.DataFrame, pd.DataFrame]:
        Xns = []
        yns = []
        for i in range(self.D):
            xd = Xd.iloc[i]
            xn, yn = self.parameters.xn(xd)
            Xns.append(xn)
            yns.append(yn)
        Xns = pd.DataFrame(data=Xns).reset_index(drop=True)
        yns = pd.DataFrame(data=yns).reset_index(drop=True)
        return Xns, yns
    # end

    # -----------------------------------------------------------------------
    # As scikit-learn estimator

    def fit(self, X, y):
        return self

    # -----------------------------------------------------------------------

    def save(self, fname):
        super().save(fname)

    def plot(self, fname):
        super().plot(fname)
# end


class MushroomClassifier:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        p = kwargs['parameters']
        tf = kwargs['target_function']
        self.parameters = p[0] if isinstance(p, list) else p
        self.target_function = tf[0] if isinstance(tf, list) else tf

    # def grid_params(self, n=None):
    #     gparams = self.parameters.grid_params(n)
    #     gparams['parameters'] = [self.parameters]
    #     gparams['target_function'] = [self.target_function]
    #     return gparams

    def get_params(self, deep=True):
        # params = self.parameters.grid_values()
        # params['parameters'] = self.parameters
        # params['target_function'] = self.target_function
        # return params
        return self.kwargs

    def fit(self, X, y):
        return self


def main():
    path("results/grid").mkdir_p()
    path("plots/grid").mkdir_p()

    data = load_data()
    X, y = data

    D = 100

    # prepare the data structures
    parameters = Parameters(data, D)
    target_function = TargetFunction(data, D)

    # prepare the estimator's parameters
    estimator_params = parameters.grid_values()
    estimator_params['target_function'] = target_function
    estimator_params['parameters'] = parameters

    # create theestimator
    estimator = MushroomClassifier(
        **estimator_params
    )

    # prepare the grid parameters
    grid_params = parameters.grid_params(5)
    grid_params['target_function'] = [target_function]
    grid_params['parameters'] = [parameters]

    # create the grid search engine
    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=grid_params,
        scoring=accuracy_score,
        n_jobs=1
    )

    gscv.fit(X, y)

    target_function.save(f"results/grid/mushroom-distilled")
    target_function.plot(f"plots/grid/mushroom-distilled")

    pass
# end


if __name__ == '__main__':
    main()
