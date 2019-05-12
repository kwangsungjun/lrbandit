import numpy as np


def d_logit(x):
    return np.exp(-x)/(1+np.exp(-x))**2


def d_logistic(x):
    return 1/(1 + np.exp(-x))


def d_identity(x):
    return 1