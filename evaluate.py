import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

def quick_acc(train, estimator, scale=False):
	X = train.drop('accuracy_group', axis=1)._get_numeric_data()
	y = train.accuracy_group

	if scale:
		rs = RobustScaler()
		X = rs.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

	estimator.fit(X_train, y_train)
	y_pred = estimator.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)
	#print(f'The accuracy of {estimator.split("(")[0]} is {accuracy}')
	return estimator.split("(")[0], accuracy