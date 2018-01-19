import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Preprocessor
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
labelizer = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features=[3])

#Importing
dataset = pd.read_csv('./data/startups.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, dataset.columns.size - 1].values

#Encoding categorical data
X[:, 3] = labelizer.fit_transform(X[:, 3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X [:, 1:]

#Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fiting
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting
Y_predicted = regressor.predict(X_test)