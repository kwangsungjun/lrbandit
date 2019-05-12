import numpy as np
from .link_functions import logit
from .link_gradients import d_logit
from scipy.stats import norm as gauss

class LinearModel(object):
    """
       set R=0.0 for logistic
    """
    def __init__(self, theta, R):
        self.theta = theta
        self.R = R
        self.t = None
        self.queried_points = None
        self.observed_data = None
        self.d = None

    def logit_linear(self, x):
        theta = self.theta
        R = self.R
        if R == 0.0:
            return logit(np.dot(theta, x))
        else:
            return logit(np.dot(theta, x) + np.random.normal(0,R**2))

    def dlogit_linear(self, x):
        theta = self.theta
        return theta*d_logit(np.dot(x,theta))

    def linear(self, x):
        theta = self.theta
        return np.dot(x, theta)

    def dlinear(self, x):
        theta = self.theta
        return theta

    @staticmethod
    def logistic_loss(theta, x, y):
        inner_product = np.dot(theta, x)
        return -y*inner_product + np.log(1 + np.exp(inner_product))

    @staticmethod
    def logistic_loss_gradient(theta, x, y):
        return (1/(1 + np.exp(-np.dot(theta, x))) - y) * x

    def observe_linear_data(self, x):
        theta = self.theta
        y = np.dot(x, theta)
        observed_data = self.observed_data
        queried_points = self.queried_points
        R = self.R
        t = self.t
        if R > 0:
            y += np.random.normal(0,R**2)

        observed_data = np.concatenate((observed_data, [y]))
        queried_points = np.vstack((queried_points, x))
        t += 1

        self.observed_data = observed_data
        self.queried_points = queried_points
        self.t = t

        return True

    def observe_binary_rewards(self, x):
        observed_data = self.observed_data
        queried_points = self.queried_points
        t = self.t
        theta = self.theta
        p = 1/(1 + np.exp(-np.dot(theta, x)))

        if np.random.rand() <= p:
            y = 1.
        else:
            y = 0.

        observed_data = np.concatenate((observed_data, [y]))
        queried_points = np.vstack((queried_points, x))
        t += 1

        self.observed_data = observed_data
        self.queried_points = queried_points
        self.t = t

    def get_binary_reward(self, x):
        theta = self.theta
        p = 1/(1 + np.exp(-np.dot(theta, x)))

        if np.random.rand() <= p:
            return 1.
        else:
            return 0.

    def get_gauss_reward(self, x):
        theta = self.theta
        inner_prod = np.dot(theta, x)

        p = gauss.cdf(inner_prod)

        if np.random.rand() <= p:
            return 1.
        else:
            return 0.

    def observe_logistic_data(self, x):
        theta = self.theta
        y = np.dot(x, theta)
        observed_data = self.observed_data
        queried_points = self.queried_points
        R = self.R
        t = self.t
        if R > 0:
            y += np.random.normal(0,R**2)

        if y >= 0:
            y = 1.
        else:
            y = -1.

        observed_data = np.concatenate((observed_data, [y]))
        queried_points = np.vstack((queried_points, x))
        t += 1

        self.observed_data = observed_data
        self.queried_points = queried_points
        self.t = t

        return True

    def initialize_data(self, x0, y0):
        queried_points = x0
        observed_data = np.array([y0])
        t = 1
        d = x0.shape[0]

        self.queried_points = queried_points
        self.observed_data = observed_data
        self.t = t
        self.d = d

    def squared_loss(self, theta_hat):
        t = self.t
        loss = 0
        queried_points = self.queried_points
        observed_data = self.observed_data

        for i in range(t):
            loss += (observed_data[i] - np.dot(queried_points[i,:], theta_hat))**2

        return loss/t

    def dsquared_loss(self, theta_hat):
        t = self.t
        queried_points = self.queried_points
        observed_data = self.observed_data
        d = self.d
        grad = np.zeros(d)
        for i in range(t):
            grad -= 2*(observed_data[i] - np.dot(queried_points[i,:], theta_hat))*queried_points[i,:]

        return grad/t

    def logistic_loss_old(self, theta_hat):
        t = self.t
        loss = 0
        queried_points = self.queried_points
        observed_data = self.observed_data

        for i in range(t):
            loss += np.log(1 + np.exp(- observed_data[i] * np.dot(queried_points[i, :], theta_hat)))

        return loss/t

    def dlogistic_loss_old(self, theta_hat):
        t = self.t
        queried_points = self.queried_points
        observed_data = self.observed_data
        d = self.d
        grad = np.zeros(d)
        for i in range(t):
            grad -= (observed_data[i] / (1 + np.exp(observed_data[i] * np.dot(theta_hat, queried_points[i])))) * \
                    queried_points[i]

        return grad/t


class LabelModel(object):
    def __init__(self, init_label, labels):
        self.init_label = labels[init_label]
        self.labels = labels

    @staticmethod
    def logistic_loss(theta, x, y):
        inner_product = np.dot(theta, x)
        return -y*inner_product + np.log(1 + np.exp(inner_product))

    @staticmethod
    def logistic_loss_gradient(theta, x, y):
        return (1/(1 + np.exp(-np.dot(theta, x))) - y) * x

    def get_reward(self, idx):
        if self.labels[idx] == self.init_label:
            return 1.
        else:
            return 0.
