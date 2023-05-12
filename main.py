import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from function import *
from extraire import *

#x, y = make_regression(n_samples=200, n_features=4, noise=10)

#x = np.hstack((x**2, x))

datasets = extraire()
datasets = numpyOfdict(datasets, length=501)

x, y = purge(datasets, indesirable = [0,1,2,3,25,26,27])

x = x.astype('float64')
y = y.astype('float64')
x = normalization(x)

x = np.hstack((x, np.ones((x.shape[0], 1))))
y = y.reshape(y.shape[0], 1)

print("Dimension de X et Y")
print("X: ",x.shape)
print("Y: ",y.shape)

train(x, y, save_modele=True, visual_result=True)
