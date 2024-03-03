from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from common import *
from skopt import gp_minimize
from stdlib.timing import tprint


# from category_encoders import OneHotEncoder


def load_data():
    df = pdx.read_data("data_uci/census+income/adult.csv",
                       numeric=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
                       # onehot=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                       #         'native-country',
                       #         'income']
                       )

    X, y = pdx.xy_split(df, target='income')
    return X, y


class TargetFunction(BaseTargetFunction):
    NUMERIC = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    ONEHOT = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
              'native-country',
              # 'income'
              ]
    TARGET = 'income'

    def __init__(self, data, D, parameters=None, maximize=True):
        super().__init__(data, D, parameters=parameters, maximize=maximize)

        X = self.X
        y = self.y

        self.xenc = pdx.SequenceEncoder([
            pdx.BinHotEncoder(columns=self.ONEHOT),
            pdx.MinMaxScaler(columns=self.NUMERIC)
        ]).fit(X)
        self.yenc = pdx.BinHotEncoder(self.TARGET).fit(y)

        self.X_true = self.xenc.transform(X)
        self.y_true = self.yenc.transform(y)

        # create the Ground Truth classifier
        self.GTC = self.create_classifier(X, y)

        # best results
        self.best_score = float('-inf') if maximize else float('inf')
        self.best_model = None
        self.best_params = None
        self.best_iter = 0
        self.maximize = maximize
        self.score_history = []
        self.best_score_history = []
        self.start_time = datetime.now()

    def create_classifier(self, X, y):
        Xenc = self.xenc.transform(X)
        yenc = self.yenc.transform(y)

        c = DecisionTreeClassifier()
        c.fit(Xenc, yenc)
        return c

    def create_labels(self, X):
        Xenc = self.xenc.transform(X)
        yenc = self.GTC.predict(Xenc)
        yenc = pd.DataFrame(data=yenc.reshape(-1, 1), columns=[self.TARGET], index=X.index)
        y = self.yenc.inverse_transform(yenc)
        return y

    def __call__(self, *args, **kwargs):
        X = reshape(*args, self.X.columns)
        y = self.create_labels(X)

        model = self.create_classifier(X, y)

        y_true = self.y_true
        y_pred = model.predict(self.X_true)

        score = accuracy_score(y_true, y_pred)
        self.score_history.append(score)
        iter = len(self.score_history)

        if score > self.best_score:
            self.best_score = score
            self.best_model = model
            self.best_params = (X, y)
            self.best_iter = iter
            tprint(f"[{iter:2}] Best score: {score}")
            self.best_score_history.append({"iter": iter, "score": score})
        else:
            tprint(f"[{iter:2}] .... score: {score}")

        return score if self.maximize else (1-score)
    # end

    def save(self, fname):
        super().save(fname)
# end


def main():
    data = load_data()

    D = 100
    parameters = Parameters(data, D)
    target_function = TargetFunction(data, D, parameters=None)

    gp_bounds = parameters.bounds()
    gp_x0 = parameters.x0()

    res = gp_minimize(
        target_function,        # target function
        gp_bounds,              # dimensions
        x0=gp_x0,
        y0=None,
        acq_func="LCB",
        acq_optimizer="auto",
        n_random_starts=5,
        n_calls=100,
        n_initial_points=10,
        n_points=1000,
        n_restarts_optimizer=5,
        random_state=777,
        xi=0.01, kappa=1.96,
        noise="gaussian",
        initial_point_generator="random",
        verbose=False,
        n_jobs=8                # n of python processes to use
                                # to speedup the analysis
    )

    target_function.save(f"census_income-distilled-{D}")

    pass


if __name__ == '__main__':
    main()
