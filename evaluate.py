import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, make_scorer
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_learning_curves
kappa_scorer = make_scorer(cohen_kappa_score(),
                           weights='quadratic')

def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def quick_eval(train, estimator, scale=False, cv=False): ##TODO modify to return trained model
	X = train.drop('accuracy_group', axis=1)._get_numeric_data()
	y = train.accuracy_group

	if scale:
		rs = RobustScaler()
		X = rs.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

	estimator.fit(X_train, y_train)
	y_pred = estimator.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	qwk = qwk3(y_test, y_pred)
	report = classification_report(y_test, y_pred)

	if cv:
		cv_score = cross_val_score(estimator, X, y, scoring='kappa_scorer', cv=5).mean()
		print(f'The CV score of {str(estimator).split("(")[0]} is {cv_score}')
		#return str(estimator).split("(")[0], cv_score
		return estimator

	else:
		print(f'The accuracy of {str(estimator).split("(")[0]} is {accuracy}')
		print(f'The QWK of {str(estimator).split("(")[0]} is {qwk}')
		print(report)
		#return str(estimator).split("(")[0], accuracy
		return estimator

