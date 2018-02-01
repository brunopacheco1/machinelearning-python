# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Importing
dataset = pd.read_csv('./data/position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fiting Linear Regression
regressor = LinearRegression()
regressor.fit(X, y)

#Predicting Linear Regression
y_predicted = regressor.predict(X)

#Fiting Polynomial Regression
#Increase degrees to fit better with the training set
poly_featurer = PolynomialFeatures(degree = 4)
X_poly = poly_featurer.fit_transform(X)
poly_featurer.fit(X_poly, y)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)

#To improve the line visualization
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

#Predicting Polynomial Regression
y_poly_predicted = poly_regressor.predict(poly_featurer.fit_transform(X_grid))

#Predicting value of a fictional level 6.5
level = 6.5
level_6_5_linear_predict = regressor.predict(level)
level_6_5_polynomial_predict = poly_regressor.predict(poly_featurer.fit_transform(level))

#Visualizing
plt.scatter(X, y, color = 'red')
plt.scatter(level, level_6_5_linear_predict, color = 'black')
plt.scatter(level, level_6_5_polynomial_predict, color = 'blue')
plt.plot(X, y_predicted, color = 'yellow')
plt.plot(X_grid, y_poly_predicted, color = 'green')
plt.title('Position vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()