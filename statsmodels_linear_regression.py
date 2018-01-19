import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

#Importing
dataset = pd.read_csv('./data/startups.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, dataset.columns.size - 1].values

"""
Learn more about this framework
http://nbviewer.jupyter.org/urls/s3.amazonaws.com/datarobotblog/notebooks/multiple_regression_in_python.ipynb
"""