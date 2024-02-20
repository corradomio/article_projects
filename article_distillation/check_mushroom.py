import pandasx as pdx
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    df = pdx.read_data("data_uci/mushroom/mushroom.csv",
                       binhot=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                               'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                               'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                               'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])

    X, y = pdx.xy_split(df, target='target')
    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=.25)

    c = DecisionTreeClassifier()
    c.fit(X_train, y_train)

    y_pred = c.predict(X_test)
    y_test = y_test.to_numpy().reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    print("accuracy:", acc)
    pass


if __name__ == '__main__':
    main()
