
"""

Simple demonstration of the look ahead estimator:  Code to compute the
stationary distribution of an AR1 model X' = beta + alpha * X + W, where W is
standard normal.  This computation is trivial, but it gives the idea.

John Stachurski, Nov 2012

"""

import numpy as np
from matplotlib import pyplot as plt

alpha, beta = 0.8, 1.0
n = 10**4

# Update rule
def g(x):
    return beta + alpha * x  

# Density of the shock
def phi(z): 
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(- (z**2) * 0.5)

# Density kernel of the model
def p(x, y):
    return phi(y - g(x))

# Generate vector of samples.
X = np.zeros(n)
W = np.random.randn(n)
for t in range(n-1):
    X[t+1] = g(X[t]) + W[t]

grid = np.linspace(-5, 15, 200)
yvals = [np.mean(p(X, y)) for y in grid]
plt.plot(grid, yvals)
plt.show()

