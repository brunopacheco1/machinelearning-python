# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#Importing
dataset = pd.read_csv('./data/position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fiting
regressor = RandomForestRegressor(n_estimators=20000, random_state=0)
regressor.fit(X, y)

#To improve the line visualization
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))

#Predicting
y_predicted = regressor.predict(X_grid)

#Predicting value of a fictional level 6.5
level = 6.5
predict = regressor.predict(level)

#Visualizing
plt.scatter(X, y, color = 'red')
plt.scatter(level, predict, color = 'black')
plt.plot(X_grid, y_predicted, color = 'blue')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()