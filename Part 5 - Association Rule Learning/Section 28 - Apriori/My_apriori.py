#Apriori

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Here we hav to do header = None because initially it was using first dataset as 
heading and we don't want that'''
# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

'''Apriori need dataset as a list of list containing each dataframe'''
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)])

'''Here we have to decide the min_support and confidence .For this case we have considered item
which is purchased at least 3 times a day so 21 times a week so support = 21/7500 rounded to 0.003
if it is not satisfactory then we can change it accordingly.'''

'''Very high confidence may lead to wrong result as the items in basket may not be dues to association but 
because they were the most purchased item so confidence should be choosed wisely. here it is 0.2'''

'''combination of .003 support and .2 confidence is good'''
#Training apriori on dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualising
results = list(rules)

# This function takes as argument your results list and return a tuple list with the format:
# [(rh, lh, support, confidence, lift)] 
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(results),
                columns=['rhs','lhs','support','confidence','lift'])