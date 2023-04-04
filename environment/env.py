from sklearn.preprocessing import PolynomialFeatures
import joblib
import numpy as np
from functools import wraps  

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

class env:
    def __init__(self, dataset, age, sex, demographic_features) -> None:
        self.ds = dataset
        self.age = age
        self.sex = sex
        self.demographic_features = demographic_features
        self.income_part = {
            'deterministic_model': joblib.load('./datasets/model/income.pkl'),
            'feature_trans': PolynomialFeatures(degree=3),
            'stochastic_param': self.ds.get_income_param(),
            'gmm': GMM(self.ds.get_income_param()),
            'last_permanent_shock': 0
        }
    

    def evlove_income(self):
        def gen_shock():
            self.income_part['last_permanent_shock'] = \
             self.income_part['stochastic_param']['persistent_param'] * self.income_part['last_permanent_shock'] + self.income_part['gmm'].sample()
            return self.income_part['last_permanent_shock'] + \
                np.random.normal(self.income_part['stochastic_param']['transitory_shock'][0], self.income_part['stochastic_param']['transitory_shock'][1])
        
        if np.random.uniform(0, 1) < self.ds.get_unemployment_rate(self.sex, self.age):
            self.income_part['last_permanent_shock'] = \
             self.income_part['stochastic_param']['persistent_param'] * self.income_part['last_permanent_shock'] + self.income_part['gmm'].sample()
            
            return 0
        self.demographic_features[0] = self.age
        main_part = np.abs(self.income_part['deterministic_model'].predict(self.income_part['feature_trans'].fit_transform([self.demographic_features])))
        return (main_part * np.exp(gen_shock())) [0]