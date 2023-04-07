from sklearn.preprocessing import PolynomialFeatures
import joblib
import numpy as np
from functools import wraps  
from lib import GMM
# from 
def add_to_list(keys):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            res = func(*args, **kw)
            for index, item in enumerate(keys):
                print(item, index)
                args[0][item].append(res[index])
            return res
        return wrapper
    return decorator

class env:
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
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
        self.employ_state = []
        self.income_state = []
        self.pension_benefit_param = {
            'fixed_rate': 0.8,
            'ave_income': [],
            'basic_pension_rate': 0.2,
            'employer_rate': 0.2,
            'exp_life': 75,
            'pension_fund_rate': 0.0275,
            'balance_basic_account': 0,
            'retire_age': 60,
            'ave_p': 1,
            'ave_stochastic_param': [0.0592753156445290, 0.018054842355693046],
            'ave_t0': 95397
        }


    @add_to_list(['income_state', 'employ_state'])
    def evlove_income(self):
        def gen_shock():
            self.income_part['last_permanent_shock'] = \
             self.income_part['stochastic_param']['persistent_param'] * self.income_part['last_permanent_shock'] + self.income_part['gmm'].sample()
            return self.income_part['last_permanent_shock'] + \
                np.random.normal(self.income_part['stochastic_param']['transitory_shock'][0], self.income_part['stochastic_param']['transitory_shock'][1])
        
        if np.random.uniform(0, 1) < self.ds.get_unemployment_rate(self.sex, self.age):
            self.income_part['last_permanent_shock'] = \
             self.income_part['stochastic_param']['persistent_param'] * self.income_part['last_permanent_shock'] + self.income_part['gmm'].sample()
            return 0, 0
        
        self.demographic_features[0] = self.age
        main_part = np.abs(self.income_part['deterministic_model'].predict(self.income_part['feature_trans'].fit_transform([self.demographic_features])))
        return (main_part * np.exp(gen_shock()))[0], 1
    
    def pension_benefit(self):
        def get_contribution_base(salary_bef, salary_now, ave_income, em_bef, em_now, fixed_rate=0.8, lb=0.6, ub=3):
            return max(min(fixed_rate*(salary_bef*em_bef + salary_now*(1-em_now)), ub*ave_income) , lb*ave_income)*em_now
        
        def evolve_ave(start, length, p, stochastic_param):
            res = [start]
            sto = np.random.normal(stochastic_param[0], stochastic_param[1], length)
            for i in range(1, length):
                res.append(res[-1] * p * np.exp(sto[i-1]))
            return res
        
        ave_list = evolve_ave(self.pension_benefit_param['ave_t0'], len(self.income_state), self.pension_benefit_param['ave_p'], self.pension_benefit_param['ave_stochastic_param'])
        print(ave_list)
        for i in range(len(self.income_state)):
            if i == 0:
                con_base = []
                con_base.append(get_contribution_base(
                    0, self.income_state[i], ave_list[i], 0, self.employ_state[i] 
                ))
                continue
            con_base.append(get_contribution_base(
                self.income_state[i-1], self.income_state[i], ave_list[i], self.employ_state[i-1], self.employ_state[i] 
            ))
        B_1 = 0.5 * (np.sum(np.array(con_base) / np.array(ave_list)) + np.sum(self.employ_state)) * 0.01 * 1 # need a param
        A_ret = np.sum(np.array(con_base) * (self.pension_benefit_param['basic_pension_rate'] + self.pension_benefit_param['employer_rate']) \
                * np.array([ pow((1+self.pension_benefit_param['pension_fund_rate']), len(con_base)-i-1) for i in range(len(con_base))]))
        
        re_age = self.pension_benefit_param['retire_age'] 
        ex_age = self.pension_benefit_param['exp_life']
        rate = self.pension_benefit_param['pension_fund_rate']
        B_2 = A_ret / ( (1-pow(1+rate, re_age - ex_age)) / (1- 1 / (1+rate)))
        return B_2 + B_1
