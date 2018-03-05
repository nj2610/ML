# Upper cofidence bound

#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

'''Initially we dont have data so first 10 ads will be selected without the algo means 
first time ad1 second time ad 2 and so on'''
#implementing ucb
import math
N = 10000
d = 10 
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if numbers_of_selections[i] > 0:
            average_rewards = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_rewards + delta_i
        else:
            upper_bound = 1e400  #1e400 = 10^4
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad]  = numbers_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
#Visualising
plt.hist(ads_selected)
plt.title('Histogrm of ad selections')
plt.xlabel('ad')
plt.ylabel('No of times ad was selected')
plt.show()