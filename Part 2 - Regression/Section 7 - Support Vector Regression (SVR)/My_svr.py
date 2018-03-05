# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

'''Before this we never need to do feature scaling but the SVR does not take care of feature scaling 
so we will have to do feature scaling by your own'''
#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

'''Here we cannot directly predict the value using 6.5 because we have applied feature scaling to
training set so we have to first aplly feature scaling on 6.5 using the sc_x.transform() but 
here also we cannot directly write 6.5 because the arguement of transform expects a array and to do 
that we use np.array() to make an array of one element and then to know the real value of output we will
have to take inverse transform'''
# Predicting a new result
Y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
Y_pred = sc_Y.inverse_transform(Y_pred)

# Visualising the SVR results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()