#Artificial neural network
#Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#REmoving first column to avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''Generally it is compulsory to apply feature scaling in ANN because it is computation heavy process 
to ease that part this step in needed.'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building ANN

#importing keras
import keras
from keras.models import Sequential    #Used to initialise neural net
from keras.layers import Dense         #Used for hidden layers

'''Two methods for initialising 1. defining as sequence of layers
2. defining as graph
here we are using sequence of layers'''
#Initialising ANN
classifier = Sequential()

'''Generally rectifier activation function is preferred for hidden layer and in output layer sigmoid activation function
is preferred as it provides probabilities'''

'''TIP: Choose number of nodes of hidden layer equal to average f number of nodes in input and out put layers.
Or we can use parameter tuning'''

'''Arguements:
    output_dim : No of nodes in hidden layer , init : weight initial,  activation: activation function used
    in first step input dim is a compulsory arguement because we have just intialised the neural net, 
    input_dim:no of independent variables'''
    
'''If dependent variable has more than two categories then activation fn is softmax for output layer'''

#Adding input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',input_dim = 11))

#Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

'''Compiling means applying stochastic gradient descent on whole ANN
Arguements :
    optimizer: adam is efficient form of stochastic gradient descent
    loss : binary for two output, for more than two categorical
    metrics : criterion for improving models performence'''
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''epochs means number of rounds
   also we have to add the batch size '''

#Fitting ANN to the dataset
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#Making Prediction and evaluating model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred >0.5)

'''Here prediction is prob but we need true false for cm'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)