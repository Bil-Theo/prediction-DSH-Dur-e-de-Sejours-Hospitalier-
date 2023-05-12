import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from function import *

x, y = make_regression(n_samples=200, n_features=1, noise=10)

x = np.hstack((x, np.ones((x.shape[0], 1))))
y = y.reshape(y.shape[0], 1)

print("Dimension de X et Y")
print("X: ",x.shape)
print("Y: ",y.shape)

train(x, y, save_modele=True, visual_result=True)
