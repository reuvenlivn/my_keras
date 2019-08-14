# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 08:40:06 2019

@author: reuve
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    #Define a “sequential” model in Keras with three layers: input, hidden and output. The
    #input layer should have layers as the number of features, for the hidden layer we will use
    #8 neurons and for the output layer we have one neuron, predicting diabetis
    #https://keras.io/getting-started/sequential-model-guide/
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add((Dense(1, activation="sigmoid")))

    # Compile the model, don’t forget to define the loss function and the optimizer as params
    # to the compile function
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

# Define random seed for reproducibility
np.random.seed(7)
#Load the indian-puma-diabetis dataset from the following link:
#http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-ind
#ians-diabetes.data
dataset = np.loadtxt('C:/ML bootcamp/ML/Naive Bayes/pima-indians-diabetes.csv', delimiter=",")

x = dataset[:,:-1]
y = dataset[:,-1]

# Define model
model = KerasClassifier(build_fn=create_model)

#Fit the model to your data, define as params the number of epochs and the batch_size
model.fit(x, y, epochs=400, batch_size=10)
#Evaluate the model on the training data usint the “evaluate” function
scores = model.model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.model.metrics_names[1], scores[1]*100))

#Predict the results on some of the training data using the “prdict” function
predictions = model.predict(x)
predictions = np.reshape(predictions,(y.size))
my_loos = (abs(y-predictions)).sum()/y.size*100
print("my score= {}%".format(my_loos))


# define hyper parameters to optimize

batch_size = [10,50,100]
epochs = [10,50]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x,y)
best_params = grid_result.best_params_

