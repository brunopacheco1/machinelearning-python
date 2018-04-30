# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

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

# Scaling
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Building the model
classifier = Sequential()
classifier.add(Dense(6, input_shape=(11,), kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dense(6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fitting
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
    
# Predicting
y_predicted = classifier.predict(X_test)

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_predicted)