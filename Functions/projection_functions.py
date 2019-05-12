import numpy as np

def unit_ball(x):
    return x/np.max(1, np.dot(x,x))