# %%

import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plot


def advance_state(Win, Wr, h, delta, x, activation_function):
    h_new = np.matmul(Win.transpose(), x) + np.matmul(Wr.transpose(), h)
    h_new = activation_function(h_new)
    delta.append(np.sum(h_new - h))
    return h_new


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(Win, Wr, W_readout, h, x):
    s = np.matmul(Win.transpose(), x) + np.matmul(Wr.transpose(), h)
    return np.matmul(W_readout.transpose(), s)


def thresh(x):
    if x >= .5:
        return 1
    return 0


V = 2  # number of features
R = 20  # size of reservoir
LEARN_TIME = 10
READOUT_TIME = 50
LEARNING_RATE = 0.001
DENSITY = 0.25
NUM_ITERATIONS = 5

Wr = random(R, R, density=DENSITY).toarray()
Win = np.random.random((V + 1, R))
W_readout = np.random.random((R, 1))

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
h = np.random.random((R, 1))
y = np.array([0, 0, 0, 1])
delta = []
old = []

for epoch in range(NUM_ITERATIONS):
    permutation = np.random.permutation(4)

    for datapoint in permutation:
        x = X[:, datapoint].reshape(2, 1)
        x = np.vstack((np.asarray(1).reshape(1,1), x))
        label = y[datapoint]

        for iteration in range(LEARN_TIME):
            h = advance_state(Win, Wr, h, delta, x, sigmoid)

        errors = []
        for iteration in range(READOUT_TIME):
            predict = forward(Win, Wr, W_readout, h, x)
            W_readout = W_readout - LEARNING_RATE * 2 * (predict - label)
            errors.append(np.square(predict - label).item(0))

        predict = forward(Win, Wr, W_readout, h, x)
        print(x[0], x[1], predict)

    print(f'Ending epoch {epoch+1}')

    old = np.asarray([Wr, Win, W_readout, h])

y = np.vstack((np.asarray(1).reshape(1,1), X[:, datapoint].reshape(2, 1)))
print(y[0], y[1], np.matmul(W_readout.transpose(), np.matmul(Win.transpose(), y) + np.matmul(Wr.transpose(), h)))

for x in range(4):
    print(np.sum(old[x] - np.asarray([Wr, Win, W_readout, h])[x]))

y = np.vstack((np.asarray(1).reshape(1,1), X[:, datapoint + 1].reshape(2, 1)))
print(y[0], y[1], np.matmul(W_readout.transpose(), np.matmul(Win.transpose(), y) + np.matmul(Wr.transpose(), h)))

for x in range(4):
    print(np.sum(old[x] - np.asarray([Wr, Win, W_readout, h])[x]))

# y = np.vstack((np.asarray(1).reshape(1,1), X[:, 1].reshape(2, 1)))
# print(y[0], y[1], np.matmul(W_readout.transpose(), np.matmul(Win.transpose(), y) + np.matmul(Wr.transpose(), h)))
#
# y = np.vstack((np.asarray(1).reshape(1,1), X[:, 2].reshape(2, 1)))
# print(y[0], y[1], np.matmul(W_readout.transpose(), np.matmul(Win.transpose(), y) + np.matmul(Wr.transpose(), h)))
#
# y = np.vstack((np.asarray(1).reshape(1,1), X[:, 3].reshape(2, 1)))
# print(y[0], y[1], np.matmul(W_readout.transpose(), np.matmul(Win.transpose(), y) + np.matmul(Wr.transpose(), h)))
