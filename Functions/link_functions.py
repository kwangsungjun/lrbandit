import numpy as np


def logit(x):
    return 1/(1. + np.exp(-x))

def identity(x):
    return x

def logistic(x):
    return np.log(1 + np.exp(x))