from numpy.random import uniform, normal
import numpy as np


class ProbDist:
    def sample(self):
        raise NotImplementedError


class Uniform2D(ProbDist):
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def sample(self):
        return np.array([uniform(-self.dim1/2, self.dim1/2), uniform(-self.dim2/2, self.dim2/2)])


class Normal2D(ProbDist):
    def __init__(self, dim1, dim2, mu, sigma):
        self.dim1 = dim1
        self.dim2 = dim2
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        point = np.array([normal(self.mu, self.sigma), normal(self.mu, self.sigma)])

        if point[0] > self.dim1/2:
            point[0] -= self.dim1
        elif point[0] < -self.dim1/2:
            point[0] += self.dim1

        if point[1] > self.dim2/2:
            point[1] -= self.dim2
        elif point[1] < -self.dim2/2:
            point[1] += self.dim2
        return point

class CorrelatedRayleighDist(ProbDist):
    def __init__(self, corr_mat):
        self.corr_mat = corr_mat

    def sample(self):
        pass