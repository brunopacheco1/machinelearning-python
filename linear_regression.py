import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing
dataset = pd.read_csv('salary.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Fiting
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting
Y_predicted = regressor.predict(X_test)

#Visualizing
plt.scatter(X_train, Y_train, color = 'red')
plt.scatter(X_test, Y_predicted, color = 'green')
plt.scatter(X_test, Y_test, color = 'black')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()