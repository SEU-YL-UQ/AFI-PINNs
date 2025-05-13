import numpy as np
import scipy.stats as stat
from pyDOE import lhs

class Uniform:
    def __init__(self, lb, ub) -> None:
        self.lb = lb 
        self.ub = ub 
        self.dim = len(ub)

    def pdf(self, samples):
        V = np.prod(self.ub - self.lb)
        return np.ones((samples.shape[0], 1))/V
    
    def sample(self, num_samples):
        return  np.random.uniform(0,1,(num_samples,self.dim))*(self.ub - self.lb) + self.lb
        
    @property
    def name(self):
        return 'uniform'


class Normal:
    def __init__(self, mu, sig) -> None:
        self.mu = mu 
        self.sig = sig 

    def pdf(self, samples):
        return stat.multivariate_normal(self.mu, self.sig).pdf(samples)
    
    def sample(self, num_samples):
        return np.random.multivariate_normal(self.mu, self.sig, num_samples)
    
    def select_sample(self, samples = None, num_samples = None):
        if samples is None:
            samples = self.sample(num_samples)
        index = np.apply_along_axis(f, 1, samples)
        return samples[~index]
    
    @property
    def name(self):
        return 'normal'

def g(t):
    x = np.cos(t) - np.cos(5*t) * np.cos(t)/4
    y = np.sin(t) - np.cos(5*t) * np.sin(t)/4
    return np.array([x, y])

def f(x):
    if x[0] == 0:
            temp1 = g(np.pi/2)
            temp2 = g(np.pi * 1.5)
            if (x[1] - temp1[1]) * (x[1] - temp2[1]) < 0:
                return True 
            else:
                return False  
    t = np.arctan(x[1]/x[0])
    temp1 = g(t)
    temp2 = g(t + np.pi)
    if (x[1] - temp1[1]) * (x[1] - temp2[1]) < 0:
        return True 
    else:
        return False




