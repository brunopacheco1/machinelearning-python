# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataSet = pd.read_csv("./data/ads_ctr_optimisation.csv")

# Implementing UCB
d = 10
N = 10000
addSelected = []
numbersOfSelections = [0] * d
sumsOfRewards = [0] * d
totalReward = 0

for n in range(0, N):
    ad = 0
    maxUpperBound = 0
    for i in range(0, d):
        if numbersOfSelections[i] > 0:
            averageReward = sumsOfRewards[i] /  numbersOfSelections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbersOfSelections[i])
            upperBound = averageReward + delta_i
        else:
            upperBound = 1e400
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i
    addSelected.append(ad)
    numbersOfSelections[ad] = numbersOfSelections[ad] + 1
    reward = dataSet.values[n, ad]
    sumsOfRewards[ad] = sumsOfRewards[ad] + reward
    totalReward = totalReward + reward
    
# Visualising the results
plt.hist(addSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()