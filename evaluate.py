import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer
from sklearn.preprocessing import RobustScaler

import mlflow, mlflow.sklearn
from collections import Counter
import joblib

#cols_to_drop = ["game_session", "installation_id", "accuracy_group"]
cols_to_drop = ['accuracy_group']

features = joblib.load("features.pkl")


def quick_eval(train, estimator, scale=False, cv=False, pc=False):

    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")

    estimator_name = str(estimator).split("(")[0]

    with mlflow.start_run(nested=True):
        from datetime import datetime

        start = datetime.now()
        features = joblib.load("features.pkl")
        X = train.drop(cols_to_drop, axis=1)
        y = train.accuracy_group

        if scale:
            rs = RobustScaler()
            X = rs.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        if pc:
            y_pred = get_class_pred(y_pred, train)
        runtime = datetime.now() - start
        accuracy = accuracy_score(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        # report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("Runtime min", runtime.seconds / 60)
        mlflow.log_param("features_shape", X.shape)
        mlflow.log_param("estimator", estimator_name)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("QWK", qwk)
        mlflow.sklearn.log_model(estimator, "model")

        if cv:
            cv_score = cross_val_score(
                estimator, X, y, scoring=kappa_scorer, cv=5
            ).mean()
            print(f"The CV qwk score of {estimator_name} is {cv_score}")
            mlflow.log_metric("CV qwk", cv_score)

            return estimator

        else:
            print(f"The accuracy of {estimator_name} is {accuracy}")
            print(f"The QWK of {estimator_name} is {qwk}")
            # print(report)

            return estimator


# get prediction
def get_class_pred(pred, train_t):
    """
    Fast cappa eval function for lgb.
    """
    dist = Counter(train_t["accuracy_group"])
    for k in dist:
        dist[k] /= len(train_t)

    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, pred)))

    return y_pred


def get_class_pred_cjs(pred, train_t):
    """
    Fast cappa eval function for lgb.
    """
    dist = Counter(train_t["accuracy_group"])
    for k in dist:
        dist[k] /= len(train_t)

    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(pred, acum * 100)

    def classify(x):
        if x <= (bound[0] * 1.1):
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2] * 0.9:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, pred)))

    return y_pred


def cv_reg(estimator, train_df, n_splits=7):
    """ Function to run cappa cross val on a given estimator"""

    with mlflow.start_run():
        results = []
        kf = KFold(n_splits=n_splits)
        i = 1
        for i, (train_idx, test_idx) in enumerate(kf.split(train_df.index)):
            train, test = train_df.iloc[train_idx], train_df.iloc[test_idx]
            X_train, y_train = train.drop(cols_to_drop, axis=1), train.accuracy_group
            X_test, y_test = test.drop(cols_to_drop, axis=1), test.accuracy_group
            estimator.fit(X_train, y_train)
            prediction = estimator.predict(X_test)
            y_pred = get_class_pred(prediction, train_df)
            qwk = cohen_kappa_score(y_pred, y_test, weights="quadratic")
            print(f"{i + 1}/{kf.n_splits} Fold start| QWK={qwk}")
            results.append(qwk)

        mlflow.log_metric("CV qwk", np.mean(results))
        mlflow.log_metric("QWK", np.max(results))
        mlflow.log_param("estimator", estimator)
        mlflow.set_tag("features", "balanced")

    return


def make_submission(preds, train_df):
    preds = get_class_pred(preds, train_df)
    # assert len(preds)==1000
    sample = pd.read_csv("data/sample_submission.csv")
    submission = pd.DataFrame()
    submission["installation_id"] = sample["installation_id"]
    submission["accuracy_group"] = preds
    submission.to_csv("preds.csv", index=False)
    return submission
