import numpy as np

class GMM():
    def __init__(self, params) -> None:
        self.mu1 = params['gmm_mu_1']
        self.mu2 = params['gmm_mu_2']
        self.std1 = params['gmm_std_1']
        self.std2 = params['gmm_std_2']
        self.p1 = params['p1']
        self.p2 = 1 - params['p1'] # unused

    def samples(self, num):
        r = np.random.uniform(0, 1, num)
        ind1 = r<=self.p1
        ind2 = r>self.p1
        len1 = len(r[ind1])
        r[ind1] = np.random.normal(self.mu1, self.std1, len1)
        r[ind2] = np.random.normal(self.mu2, self.std2, num - len1)
        return r
    
    def sample(self):
        return np.random.normal(self.mu1, self.std1) if np.random.uniform(0, 1) <= self.p1 else np.random.normal(self.mu2, self.std2)
