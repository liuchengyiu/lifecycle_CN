import pandas as pd
import numpy as np
import time
class DataSet:
    def __init__(self) -> None:
        self.create_time = time.time()
    
    def get_unemployment_rate(self, sex, age):
        # sex: {0, 1} 0 male, 1 female
        # age 16~65
        if sex not in [0, 1] and (age < 16 or age > 65):
            raise Exception('Error: illegal param for unemployment rate.')
        if hasattr(self, '_unemployment_rate'):
            return self._unemployment_rate[sex][age-16]
        _unemployment_rate = []
        weights = [0.1, 0.1, 0.1, 0.3, 0.4]
        year = [2011, 2013, 2015, 2017, 2019]
        df = pd.read_csv('./datasets/unemployment_rate.csv')
        for n_sex in ['male', 'female']:
            res = [0]*len(df)
            for i in range(len(weights)):
                now = (df['{}_unemployed_{}'.format(n_sex, year[i])] / (df['{}_unemployed_{}'.format(n_sex, year[i])] + df['{}_employed_{}'.format(n_sex, year[i])]) ).tolist()
                now = [x*weights[i] for x in now]
                res = np.sum([res, now],axis=0).tolist()
            _unemployment_rate.append(res[::-1])
        self._unemployment_rate = _unemployment_rate
        return self._unemployment_rate[sex][age-16]

        