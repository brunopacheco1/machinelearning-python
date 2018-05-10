# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

dataSet = pd.read_csv("./data/churn_modelling.csv")
X = dataSet.iloc[:, 3:13].values
y = dataSet.iloc[:, 13].values

# Encoding categorical data

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting
y_predicted = classifier.predict(X_test)

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_predicted)

# K-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
accuracies.mean()
accuracies.std()