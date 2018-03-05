'''Problem was that a man went to a company and told that he has 6.5 level of job and 
he was earning 160000 dollars we have to find out that whether he is telling truth or not'''

#Polynomial regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

'''
Here we should not split the data set into training data and test data because we have only 
10 observation and we should train our model with as much observation as we can
'''

#Fitting Linear regression to our training set
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

'''
This polynomial features automatically adds the x0=1 column in the beginning of
the feature matrix and also adds all the clumns upto the given degree in the end
of the feature matrix and then we fit the model on new X_poly also we have to keep on 
increasing the value of degree until we get the best fit.
'''

#Fitting Polynomial regression to our training data
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# Visualising the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''Here we put poly_reg.fit_transform(X) in the brackets instead of X_poly because the X_poly is 
already defined for a particular X if we have to change observations we will have to change only X '''
# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''Since the above curve was looking like collection of straight lines between unit steps so to 
obtain a smooth curve we make the steps smaller'''
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
#%matplotlib auto
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(np.array([[6.5]])))
