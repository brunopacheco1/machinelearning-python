import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Importing
dataset = pd.read_csv('./data/startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, dataset.columns.size - 1].values

#Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)