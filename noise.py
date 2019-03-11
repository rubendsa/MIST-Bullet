import numpy as np 

class Noise():
    NO_NOISE = 0
    NOISE = 2

    def __init__(self, noise_type):
        self.noise_type = noise_type 
        self.cov = 0 * np.identity(4)
        self.chol = 0 * np.identity(4)

    def get_noise(self):
        if self.noise_type == self.NO_NOISE:
            return np.zeros(4)
        else:
            noise_v = np.random.ranf(4) #TODO decide if [0,1) is satisfactory or need normal dist.
            return np.matmul(self.chol, noise_v)
    
    def get_covariance(self):
        return self.cov
    
    def set_cov(self, new_cov):
        self.cov = new_cov * np.identity(4)
        self.chol = np.linalg.cholesky(self.cov)