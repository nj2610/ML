#Importing Libraries
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
Dataset = pd.read_csv('Data.csv')
X = Dataset.iloc[:, :-1].values
Y = Dataset.iloc[:, 3].values



#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset to training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
'''Here we don't need to feature scale our Y set because it is already in the same range X 
as it is categorical data but if we encounter regression problem in which the dependent 
variables are also varying in large range then we will need to feature scale that also '''