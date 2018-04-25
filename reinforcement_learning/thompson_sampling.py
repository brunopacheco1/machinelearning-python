# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataSet = pd.read_csv("./data/ads_ctr_optimisation.csv")

# Implementing Thompson Sampling
d = 10
N = 10000
addSelected = []
numbersOfRewards_1 = [0] * d
numbersOfRewards_0 = [0] * d
totalReward = 0

for n in range(0, N):
    ad = 0
    maxRandom = 0
    for i in range(0, d):
        randomBeta = random.betavariate(numbersOfRewards_1[i] + 1, numbersOfRewards_0[i] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i
    addSelected.append(ad)
    reward = dataSet.values[n, ad]
    totalReward = totalReward + reward
    if(reward == 1): numbersOfRewards_1[ad] = numbersOfRewards_1[ad] + 1
    else: numbersOfRewards_0[ad] = numbersOfRewards_0[ad] + 1
    
# Visualising the results
plt.hist(addSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()