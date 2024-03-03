from datetime import datetime
import pandas as pd
import pandasx as pdx
import jsonx as jsx


class BaseTargetFunction:
    def __init__(self, data, D, maximize=True):
        X, y = data
        self.X = X  # features
        self.y = y  # target
        self.D = D  # n of distilled points
        self.M = X.shape[1]  # n of features

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
        def duration():
            seconds = (self.done_time - self.start_time).total_seconds()
            if seconds <= 60:
                return f"{seconds} s"
            if seconds <= 3600:
                minutes = seconds // 60;
                seconds = seconds % 60;
                return f"{minutes:02}:{seconds:02}"
            else:
                hours = seconds // 3600
                minutes = seconds % 3600 // 60
                seconds = seconds % 60
                return f"{hours:02}:{minutes:02}:{seconds:02}"
        # end
        self.done_time = datetime.now()

        df = pd.concat(self.best_params, axis=1)
        pdx.save(df, fname+".csv", index=False)
        jsx.save({
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "done_time": self.done_time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": delta_time(self.start_time, self.done_time),

            "n_iter": len(self.score_history),
            "n_distilled_points": self.D,
            "n_features": self.M,
            "n_targets": self.y.shape[1],

            "classifier": self.best_model.__class__.__name__,

            "best_score": {"iter": self.best_iter, "score": self.best_score},
            "score_history": self.score_history,
            "best_score_history": self.best_score_history
        }, fname+".json")
        pass



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

