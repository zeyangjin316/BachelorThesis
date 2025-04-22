import numpy as np
import scipy.stats as stat
from copulas.multivariate import GaussianMultivariate

class SkewedTCopula:
    def __init__(self, nu, lambda_):
        self.nu = nu    #Degrees of freedom
        self.lambda_ = lambda_  #Skewness parameter

    def cdf(self, u1, u2):
        return 


