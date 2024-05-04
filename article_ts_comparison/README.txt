Problem
    1) TS
        univariate/multivariate
        whithout/with input features
    2) File containing
        single TS
        multiple TS based on 1+ grouping columns

TS models:
    1) statistical models
        ARMA like
        GARCH like

    2) TS models based on ML models
        Decision Tree
        Linear Regression
        Neural Network
            RNN (LSTM, ...)
            Transformer
            other models

    3)


TS prediction
    1) recursive
        1-step: one timeslot after the other
        fh-steps: forecasting horizon steps in a single execution
    2) fh-steps
        the model is able to predict fh steps in a single prediction
        there is a different model for each timeslot in the forecasting horizon

