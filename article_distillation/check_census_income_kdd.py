import pandas as pd
import pandasx as pdx
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from timing import tprint


def main():
    # df = pd.read_csv("data_uci/census+income+kdd/census-income.csv", na_values=['?', 'Unk', 'NA'])

    df = pdx.read_data("data_uci/census+income+kdd/census-income.csv",
                       numeric=['age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks',
                                'weeks worked in year', 'instance weight'],
                       onehot=['class of worker', 'education', 'enroll in edu inst last wk', 'marital stat',
                               'major industry code', 'major occupation code', 'race', 'hispanic origin', 'sex',
                               'member of a labor union', 'reason for unemployment',
                               'full or part time employment stat', 'tax filer stat', 'region of previous residence',
                               'state of previous residence', 'detailed household and family stat',
                               'detailed household summary in household',
                               'family members under 18', 'country of birth father', 'country of birth mother',
                               'country of birth self', 'citizenship', 'fill inc questionnaire for veteran\'s admin',
                               'veterans benefits', 'own business or self employed', 'num persons worked for employer',
                               'income'
                               ],
                       ignore=['detailed industry recode', 'detailed occupation recode',
                               'migration code-change in msa', 'migration code-change in reg',
                               'migration code-move within reg', 'migration prev res in sunbelt',
                               'live in this house 1 year ago', 'year'],
                       na_values=['?', 'Unk']
                       )

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
