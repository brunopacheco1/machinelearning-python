# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing
dataset = pd.read_csv('./data/salary.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fiting
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting
y_predicted = regressor.predict(X_test)

#Visualizing
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_predicted, color = 'green')
plt.scatter(X_test, y_test, color = 'black')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()