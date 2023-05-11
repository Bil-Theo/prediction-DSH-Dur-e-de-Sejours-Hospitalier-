import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=200, n_features=1, noise=10)

y = y.reshape(y.shape[0], 1)

print("X: ",x.shape)
print("Y: ",y.shape)