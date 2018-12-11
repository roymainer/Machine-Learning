"""
XOR gate via machine learning

XOR:    A B y
        0 0 0
        0 1 1
        1 0 1
        1 1 0
"""


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

np.random.seed(444)  # initialize random seed to reproduce the same results when running the program again

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # X array containing the 4 possible A-B sets of input to the XOR gate
y = np.array([[0], [1], [1], [0]])

model = Sequential()  # neural network with sequential layers
model.add(Dense(2, input_dim=2))  # first layer of neurons composed of two neurons fed by two inputs
model.add(Activation('sigmoid'))  # their activation function is a sigmoid function
model.add(Dense(1))  # output layer is composed of a single neuron
model.add(Activation('sigmoid'))  # with the same activation function

sgd = SGD(lr=0.1)  # Stochastic Gradient Descent will adjust the weights of the network with learning rate eq. 0.1
model.compile(loss="mean_squared_error", optimizer=sgd)  # "MSE" will be a the loss function to be minimized

# run training using X and y as training examples, updating the weights after every training example is fed into
# the network (batch_size=1). The number of epoches represents the number of times the whole training set will be used
# to train the neural network (5000 * 4)
model.fit(X, y, batch_size=1, epochs=5000)

if __name__ == '__main__':
    print(model.predict(X))
