import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_learning_curves


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

	if cv:
		cv_score = cross_val_score(estimator, X, y, cv=5).mean()
		print(f'The CV score of {str(estimator).split("(")[0]} is {cv_score}')
		return str(estimator).split("(")[0], cv_score
	else:
		print(f'The accuracy of {str(estimator).split("(")[0]} is {accuracy}')

		return str(estimator).split("(")[0], accuracy

