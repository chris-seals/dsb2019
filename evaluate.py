import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer
from sklearn.preprocessing import RobustScaler
import sys, os
sys.path.append(os.getcwd())
from prepare import balance_classes
import mlflow, mlflow.sklearn
from collections import Counter
import joblib

cols_to_drop = ["game_session", "installation_id", "accuracy_group"]
#cols_to_drop = ['accuracy_group']

features = joblib.load("features.pkl")


def quick_eval(train, estimator, scale=False, cv=False, pc=False):

    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")

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
        X_train, y_train = balance_classes(X_train, y_train)
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
        mlflow.log_param("estimator", estimator.__class__.__name__)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("QWK", qwk)
        mlflow.sklearn.log_model(estimator, "model")
        mlflow.log_params(estimator.get_params())
        mlflow.set_tag(key="features", value="balanced")

        if cv:
            cv_score = cross_val_score(
                estimator, X, y, scoring=kappa_scorer, cv=5
            ).mean()
            print(f"The CV qwk score of {estimator.__class__.__name__} is {cv_score}")
            mlflow.log_metric("CV qwk", cv_score)

            return estimator

        else:
            print(f"The accuracy of {estimator.__class__.__name__} is {accuracy}")
            print(f"The QWK of {estimator.__class__.__name__} is {qwk}")
            # print(report)

            return estimator


# get prediction
def get_class_pred(pred, train_t):
    """
    Fast cappa eval function for regression outputs
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


# def get_class_pred(preds, X_test):
#     test = X_test
#     test['accuracy_group'] = preds
#     #print(test)
#     #print(test.accuracy_group)
#
#     """Example
#     {8: {0: 0.9081617833774993, 1: 1.3557792939869913, 2: 1.6557723302312697},
#      15: {0: 0.750632477134402, 1: 1.3367320716399504, 2: 1.71574333163995},
#      16: {0: 1.0165306932685654, 1: 1.4530646993771625, 2: 1.7328451756800636},
#      20: {0: 2.078115677309628, 1: 2.301674688587722, 2: 2.4390138805736328},
#      40: {0: 1.6276847459431902, 1: 2.0950951087691188, 2: 2.306488824806992}}
#      """
#     raw_sd = {8: {0: 576, 1: 353, 2: 470, 3: 2752},
#               15: {0: 421, 1: 459, 2: 630, 3: 2545},
#               16: {0: 594, 1: 355, 2: 460, 3: 2348},
#               20: {0: 1752, 1: 466, 2: 256, 3: 507},
#               40: {0: 886, 1: 778, 2: 389, 3: 693}}
#
#     # establish separate bounds for each assessment type (8,15,16,20,40) and assemble master distribution dictionary
#     master = {}
#     for session in raw_sd:
#         session_array = X_test[X_test.session_title == session]['accuracy_group']
#         dist = raw_sd[session]
#
#         sum_values = sum(dist.values())
#         for k in dist:
#             dist[k] /= sum_values
#         acum = 0
#         bound = {}
#         for i in range(3):
#             acum += dist[i]
#             bound[i] = np.percentile(session_array, acum * 100)
#         master[session] = bound
#
#     def classify(row):
#         sid = row['session_title']
#         x = row['accuracy_group']
#         #print(sid,x)
#         if x <= master[sid][0]:
#             return 0
#         elif x <= master[sid][1]:
#             return 1
#         elif x <= master[sid][2]:
#             return 2
#         else:
#             return 3
#
#     X_test['accuracy_group'] = X_test.apply(classify, axis=1)
#
#     return test['accuracy_group']


def cv_reg(estimator, train_df, n_splits=7):
    """ Function to run cappa cross val on a given estimator"""
    estimator_name = str(estimator).split("(")[0]
    with mlflow.start_run():
        results = []
        kf = KFold(shuffle=True, n_splits=n_splits)
        i = 1
        for i, (train_idx, test_idx) in enumerate(kf.split(train_df.index)):
            train, test = train_df.iloc[train_idx], train_df.iloc[test_idx]
            X_train, y_train = train.drop(cols_to_drop, axis=1), train.accuracy_group
            # try balancing
            X_train, y_train = balance_classes(X_train, y_train)
            X_test, y_test = test.drop(cols_to_drop, axis=1), test.accuracy_group
            estimator.fit(X_train, y_train)
            prediction = estimator.predict(X_test)
            y_pred = get_class_pred(prediction, train_df)
            qwk = cohen_kappa_score(y_pred, y_test, weights="quadratic")
            print(f"{i + 1}/{kf.n_splits} Fold start| QWK={qwk}")
            results.append(qwk)
        mlflow.log_params(estimator.get_params())
        mlflow.log_metric("CV qwk", np.mean(results))
        mlflow.log_metric("QWK", np.max(results))
        mlflow.log_param('train_shape', train_df.shape)
        mlflow.log_param("estimator", estimator.__class__.__name__)
        mlflow.set_tag(key="features", value="balanced")

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
