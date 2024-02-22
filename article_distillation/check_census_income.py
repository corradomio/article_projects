import pandasx as pdx
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from timing import tprint


def main():
    df = pdx.read_data("data_uci/census+income/adult.csv",
                       numeric=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
                       onehot=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                               'native-country',
                               'income'])

    tprint("Split data")
    X, y = pdx.xy_split(df, target='income')
    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=.25)

    tprint("Train model")
    c = DecisionTreeClassifier()
    c.fit(X_train, y_train)

    tprint("Predictions")
    y_pred = c.predict(X_test)
    y_test = y_test.to_numpy().reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    tprint("accuracy:", acc)
    pass


if __name__ == '__main__':
    main()
