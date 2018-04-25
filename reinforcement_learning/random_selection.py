# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataSet = pd.read_csv("./data/ads_ctr_optimisation.csv")

# Implementing Random Selection
import random
N = 10000
d = 10
addSelected = []
totalReward = 0
for n in range(0, N):
    ad = random.randrange(d)
    addSelected.append(ad)
    reward = dataSet.values[n, ad]
    totalReward = totalReward + reward

# Visualising the results
plt.hist(addSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()