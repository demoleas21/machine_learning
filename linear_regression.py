import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = np.genfromtxt('C:\Users\Andrew\Downloads\data.csv', delimiter=',')

X = data[:, 0].reshape(-1, 1)
Y = data[:, 1]
model = LinearRegression()
model.fit(X, Y)

plt.scatter(X, Y)  # generate a sctter plot from X and Y

print(model.coef_)
print(model.intercept_)

plt.plot(X, model.predict(X))
plt.show()

