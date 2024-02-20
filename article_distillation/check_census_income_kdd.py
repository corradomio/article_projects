import pandasx as pdx
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    df = pdx.read_data("data_uci/census+income+kdd/census-income.csv",
                       numeric=['age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks',
                                'weeks worked in year'],
                       onehot=['class of worker', 'education', 'enroll in edu inst last wk', 'marital stat',
                               'major industry code', 'major occupation code', 'race', 'hispanic origin', 'sex',
                               'member of a labor union', 'reason for unemployment',
                               'full or part time employment stat', 'tax filer stat', 'region of previous residence',
                               'state of previous residence', 'detailed household and family stat',
                               'detailed household summary in household', 'migration code-change in msa',
                               'migration code-change in reg', 'migration code-move within reg',
                               'live in this house 1 year ago', 'migration prev res in sunbelt',
                               'family members under 18', 'country of birth father', 'country of birth mother',
                               'country of birth self', 'citizenship', 'fill inc questionnaire for veteran\'s admin',
                               'veterans benefits', 'own business or self employed', 'num persons worked for employer',
                               'income'
                               ],
                       ignore=['detailed industry recode', 'detailed occupation recode', 'instance weight']
                       )

    X, y = pdx.xy_split(df, target='income')
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
