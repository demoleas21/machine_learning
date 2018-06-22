import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = np.genfromtxt('C:\Users\Andrew\Downloads\data.csv', delimiter=',')

X = data[:, 0].reshape(-1, 1)
Y = data[:, 1]


def compute_error(X, Y, w0, w1):
    total_error = 0
    M = float(len(X))
    for idx, _ in enumerate(X):
        x_i, y_i = X[idx], Y[idx]
        y_cap = w1 * x_i + w0
        total_error += (1/M) * ((y_cap - y_i) ** 2)
    return total_error

def gradient_step(X, Y, learning_rate, w0, w1):
    assert len(X) == len(Y)
    M = float(len(X))
    der_w0, der_w1 = 0, 0
    for idx, _ in enumerate(X):
        x_i = X[idx]
        y_i = Y[idx]
        y_cap = w1 * x_i + w0
        der_w0 += (-2/M) * (y_i - y_cap)
        der_w1 += (-2/M) * (y_i - y_cap) * x_i
    w0_new = w0 - der_w0 * learning_rate
    w1_new = w1 - der_w1 * learning_rate
    return w0_new, w1_new


def gradient_runner(X, Y, init_w0, init_w1, learning_rate, epochs):
    w0, w1 = init_w0, init_w1
    for epoch in range(epochs):
        w0, w1 = gradient_step(X, Y, learning_rate, w0, w1)  # the outputed w0, w1 are the values after a batch step
        print('After epoch# %s, the error was: %s' % (epoch, compute_error(X, Y, w0, w1)))
    return w0, w1


if __name__ == '__main__':
    learning_rate = 0.001
    initial_w0, initial_w1 = 0, 0
    epochs = 100
    final_w0, final_w1 = gradient_runner(X, Y, initial_w0, initial_w1, learning_rate, epochs)
    print(final_w0, final_w1)
