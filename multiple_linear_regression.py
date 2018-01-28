import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Preprocessor
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
labelizer = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features=[3])

#Importing
dataset = pd.read_csv('./data/startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, dataset.columns.size - 1].values

#Encoding categorical data
X[:, 3] = labelizer.fit_transform(X[:, 3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X [:, 1:]

#Optimal result provided by backward elimination
X = X[:, [2]]

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

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
plt.title('R&D Spend vs Profit')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

"""
#Building optimal using Backward Elimination
X = np.append(np.ones((50, 1)).astype(int), X, 1)

variables = [0, 1, 2, 3, 4, 5]

while True:
    not_found = True    
    X_optimal = X[:, variables]
    regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
    new_variables = []
    index = 0;
    highest_p_value = 0.05
    highest_p_index = 0
    for pvalue in regressor_OLS.pvalues:
        if pvalue > highest_p_value:
            not_found = False
            highest_p_value = pvalue
            highest_p_index = index
        index = index + 1
        
    if not_found:
        break
    else:
        for old_index in range(0, len(variables)):
            if old_index != highest_p_index:
                new_variables.append(variables[old_index])
        variables = new_variables
print(variables)
"""