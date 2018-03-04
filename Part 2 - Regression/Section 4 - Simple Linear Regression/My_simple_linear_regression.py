# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib auto 
#To show graphs in separate window
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3.0, random_state = 0)

#fitting the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the values for test set 
Y_pred = regressor.predict(X_test)

""" If you want to predict the salary for any particular year of experience then we can use this:
print(regressor.predict(0))"""

#Visualizing the plots for training set
plt.scatter(X_train, Y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('Salary vs Experience for Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the plots for test set
plt.scatter(X_test, Y_test, color ='red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
'''Since our regressor is trained on train set so whether we replace train by test or not we will get same result'''
plt.title('Salary vs Experience for Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
