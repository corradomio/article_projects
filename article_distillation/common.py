import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import pandasx as pdx
import jsonx as jsx
from stdlib import lrange


def reshape(l: list[str], columns: list[str]):
    m = len(columns)
    M = []
    for i in range(0, len(l), m):
        M.append(l[i:i+m])
    return pd.DataFrame(data=M, columns=columns)
# end


def delta_time(start: datetime, done: datetime):
    seconds = int((done - start).total_seconds())
    if seconds < 60:
        return f"{seconds} s"
    elif seconds < 3600:
        s = seconds % 60
        m = seconds // 60
        return f"{m:02}:{s:02} s"
    else:
        s = seconds % 60
        seconds = seconds // 60
        m = seconds % 60
        h = seconds // 60
        return f"{h:02}:{m:02}:{s:02} s"
# end


class Parameters:
    def __init__(self, data, D):
        X, y = data
        self.X: pd.DataFrame = X
        self.y: pd.DataFrame = y
        self.D: int = D
        # make the columns order 'consistent'
        self.columns = X.columns
        # categorical values
        self.column_ranges = pdx.columns_range(X)

    def bounds(self):
        columns_range = self.column_ranges
        D = self.D
        bounds = [columns_range[col].bounds() for i in range(D) for col in self.columns]
        return bounds

    def x0(self):
        columns_range = self.column_ranges
        D = self.D
        x0 = [columns_range[col].random() for i in range(D) for col in self.columns]
        return x0

    def xn(self, x1):
        """x nearest to x1, and it's label"""
        min_dist = float("inf")
        xs = None
        ys = None
        n = len(self.X)
        for i in range(n):
            x2 = self.X.iloc[i]
            dist = self.distance(x1, x2)
            if dist < min_dist:
                min_dist = dist
                xs = x2
                ys = self.y.iloc[i]
        # end
        return xs, ys

    def distance(self, x1: pd.Series, x2: pd.Series) -> float:
        """Distance between two points"""
        columns_range = self.column_ranges
        dist = 0
        for i, col in enumerate(columns_range.keys()):
            c1 = x1.iloc[i]
            c2 = x2.iloc[i]
            dist += columns_range[col].distance(c1, c2)
        return dist
    # end
# end


def nameof(s: str) -> str:
    p = s.find('\\')
    if p == -1:
        p = s.find('/')
    if p != -1:
        s = s[p+1:]
    p = s.find('-')
    if p != -1:
        s = s[:p]
    return s

class BaseTargetFunction:
    def __init__(self, data, D, parameters=None, maximize=True):
        X, y = data
        self.X = X  # features
        self.y = y  # target
        self.D = D  # n of distilled points
        self.M = X.shape[1]  # n of features

        self.parameters = parameters

        # Ground Truth Classifier
        self.GTC = None

        # best results
        self.best_score = float('-inf') if maximize else float('inf')
        self.best_model = None
        self.best_params = None
        self.best_iter = 0
        self.maximize = maximize
        self.score_history = []
        self.best_score_history = []

        self.start_time = datetime.now()
        self.done_time = datetime.now()

    # end

    def create_classifier(self, X, y):
        ...

    def create_labels(self, X):
        ...

    def make_target(self, y):
        ...

    def __call__(self, *args, **kwargs):
        ...

    def save(self, fname):
        self.done_time = datetime.now()

        D = self.D
        date_ext = self.start_time.strftime("%Y%m%d.%H%M%S")
        cname = f"{fname}-{D}-{date_ext}.csv"
        jname = f"{fname}-{D}-{date_ext}.json"

        df = pd.concat(self.best_params, axis=1)
        pdx.save(df, cname, index=False)
        jsx.save({
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "done_time": self.done_time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": delta_time(self.start_time, self.done_time),
            "synthetic_points": self.parameters is None,

            "n_iter": len(self.score_history),
            "n_distilled_points": self.D,
            "n_features": self.M,
            "n_targets": self.y.shape[1],

            "classifier": self.best_model.__class__.__name__,

            "best_score": {"iter": self.best_iter, "score": self.best_score},
            "score_history": self.score_history,
            "best_score_history": self.best_score_history
        }, jname)
        pass

    def plot(self, fname):
        D = self.D
        name = nameof(fname)

        plt.clf()

        # scores
        y = self.score_history
        x = lrange(1, len(y) + 1)

        # best scores
        bx = [bs['iter'] for bs in self.best_score_history]
        by = [bs['score'] for bs in self.best_score_history]

        plt.plot(x, y)
        plt.scatter(bx, by, c="r", s=50)
        plt.ylim((0, 1))
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.title(f"{name}, {D} points")

        date_ext = self.start_time.strftime("%Y%m%d.%H%M%S")
        pname = f"{fname}-{D}-{date_ext}.png"

        plt.savefig(pname, dpi=300)
# end
