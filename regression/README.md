# Regression Algorithms
Some examples of Regression algorithms using Python.

## Regression problem

The overall idea of regression is to examine two things: 

1. does a set of predictor variables do a good job in predicting an outcome (dependent) variable?
2. Which variables in particular are significant predictors of the outcome variable, and in what way do they–indicated by the magnitude and sign of the beta estimates–impact the outcome variable?

### Linear Regression (One or more variables)

The simplest form of the regression equation with one dependent and one or more independent variables, defined by the formula below, where y = estimated dependent variable score, c = constant, tetha = regression coefficient, and x = score on the independent variable.

![Linear Regression Equation](/images/linear_regression_equation.png)

- One Variable Linear Regression on file **linear_regression.py**;

- Multiple Variables Linear Regression on file **multiple_linear_regression.py**;

### Polynomial Regression

### Support Vector Regression (SVR)

### Decision Tree Regression

### Random Forest Regression

### Pros and cons about regression algorithms

Algorithm | Pros | Cons
------------ | ------------- | -------------
Linear Regression | Works on any size of dataset, gives informations about relevance of features | The Linear Regression Assumptions
Polynomial Regression | Works on any size of dataset, works very well on non linear problems | Need to choose the right polynomial degree for a good bias/variance tradeoff
SVR | Easily adaptable, works very well on non linear problems, not biased by outliers | Compulsory to apply feature scaling, not well known, more difficult to understand
Decision Tree Regression | Interpretability, no need for feature scaling, works on both linear / nonlinear problems | Poor results on too small datasets, overfitting can easily occur
Random Forest Regression | Powerful and accurate, good performance on many problems, including non linear | No interpretability, overfitting can easily occur, need to choose the number of trees