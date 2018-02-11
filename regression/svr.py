# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

#Importing
dataset = pd.read_csv('../data/position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Fiting SVR
#The diference is on choosing the best kernel to be used in your model
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X, y)

#To improve the line visualization
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

#Predicting SVR
y_svr_predicted = svr_regressor.predict(X_grid)

#Predicting value of a fictional level 6.5
level = scaler_X.transform(6.5)
level_6_5_svr_predict = svr_regressor.predict(level)

#Inverse scaler
X = scaler_X.inverse_transform(X)
X_grid = scaler_X.inverse_transform(X_grid)
y = scaler_y.inverse_transform(y)
level = scaler_X.inverse_transform(level)
level_6_5_svr_predict = scaler_y.inverse_transform(level_6_5_svr_predict)
y_svr_predicted = scaler_y.inverse_transform(y_svr_predicted)

#Visualizing
plt.scatter(X, y, color = 'red')
plt.scatter(level, level_6_5_svr_predict, color = 'black')
plt.plot(X_grid, y_svr_predicted, color = 'green')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()