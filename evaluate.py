import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    make_scorer,
)
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_learning_curves
import mlflow, mlflow.sklearn


def quick_eval(train, estimator, scale=False, cv=False):

    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")

    estimator_name = str(estimator).split("(")[0]

    with mlflow.start_run(nested=True):
        from datetime import datetime

        start = datetime.now()
        X = train.drop("accuracy_group", axis=1)._get_numeric_data()
        y = train.accuracy_group

        if scale:
            rs = RobustScaler()
            X = rs.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        runtime = datetime.now() - start
        accuracy = accuracy_score(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("Runtime min", runtime.seconds/60)
        mlflow.log_param("features_shape", X.shape)
        mlflow.log_param("estimator", estimator_name)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("QWK", qwk)
        mlflow.sklearn.log_model(estimator, "model")

        #classes = [str(x) for x in [0, 1, 2, 3]]

        # create a metric for each piece of the confusion matrix
        #for i in classes:
        #    [mlflow.log_metric(f"{i}-{k}", v) for k, v in report[i].items()]

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
            #print(report)

            return estimator
