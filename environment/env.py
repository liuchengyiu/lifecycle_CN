from sklearn.preprocessing import PolynomialFeatures
import joblib
import numpy as np
from functools import wraps  
from lib import GMM
from datasets import DataSet
import copy

# from 
def add_to_list(keys):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            res = func(*args, **kw)
            for index, item in enumerate(keys):
                args[0][item].append(res[index])
            return res
        return wrapper
    return decorator

class env:
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __init__(self, dataset: DataSet, age, sex, demographic_features, health_features) -> None:
        self.ds = dataset
        self.age = age
        self.init_age = age
        self.sex = sex
        self.risk_free_rate = dataset.risk_free_rate
        self.exp_life = dataset.exp_life
        self.retire_age = dataset.retire_age
        self.s_age = dataset.s_age
        self.child_num = [2]*100
        self.demographic_features = demographic_features
        self.health_features =  health_features
        self.lowest_comsumption = self.ds.lowest_comsumption
        self.income_part = {
            'deterministic_model': joblib.load('./datasets/model/income.pkl'),
            'feature_trans': PolynomialFeatures(degree=3),
            'stochastic_param': self.ds.get_income_param(),
            'gmm': GMM(self.ds.get_income_param()),
            'last_permanent_shock': 0
        }
        self.health_part = self.ds.get_medical_param(self.health_features)
        self.employ_state = []
        self.income_state = []
        self.pension_benefit_param = self.ds.get_pension_benefit_param()
        self.rate_model = self.ds.get_rate_model()
        self.house_param = self.ds.house_param()
        self.investment_asset = self.ds.get_investment_asset()
        self.reward_param = self.ds.get_reward_param()
        self.state_dim = 16
        self.action_dim = 2
        self.param_dim = 10 * self.action_dim
        

    def reset(self):
        self.__init__(self.ds, self.init_age, self.sex, self.demographic_features, self.health_features)
        income_next = self.evlove_income()[0]
        kid_num = self.child_num[self.age]
        pension_bef, con_base = self.pension_benefit()
        # health 
        health_care_cost = 0
        health_status = 1
        living_pro = 1
        if self.age >= self.retire_age:
            health_care_cost, health_status, accumulated_shock, living_pro = self.get_health_cost()
            self.health_part['accumulated_shock'] = accumulated_shock
            self.health_part['health_status'] = health_status
        else:
            health_care_cost = self.health_part['hs_fractor'] * self.health_part['g']
        # tex a1 * tex^a2,  a1 = 0.15, a2 = 0.98
        exp_tex = 12000 * kid_num
        if self.house_param['owning_house'] == 0:
            exp_tex += 12000
        if self.house_param['owning_house'] == 1 and self.house_param['debt'] > 0:
            exp_tex += 12000
        income = income_next + 0.15 * np.power(exp_tex, 0.98)
        long_term_bond, equity_return, p_long_term_bond, p_equity_return = self.investment_asset['long_term_bond'],\
            self.investment_asset['equity_return'], self.investment_asset['p_long_term_bond'], self.investment_asset['p_equity_return']
        state = [
            self.init_age, float(self.health_part['health_status']), health_care_cost, living_pro, kid_num,\
            income, con_base, self.house_param['owning_house'], self.house_param['house_fund'], long_term_bond, equity_return, p_long_term_bond, p_equity_return,\
            self.house_param['debt'] - self.house_param['house_fund_loan'], self.house_param['house_fund_loan'], self.house_param['house_price']
        ]
        return state

    # @add_to_list(['income_state', 'employ_state'])
    def evlove_income(self):
        def gen_shock():
            self.income_part['last_permanent_shock'] = \
             self.income_part['stochastic_param']['persistent_param'] * self.income_part['last_permanent_shock'] + self.income_part['gmm'].sample()
            return self.income_part['last_permanent_shock'] + \
                np.random.normal(self.income_part['stochastic_param']['transitory_shock'][0], self.income_part['stochastic_param']['transitory_shock'][1])
        if self.age < self.retire_age:
            income = 0
            hire_status = 0
            if np.random.uniform(0, 1) > self.ds.get_unemployment_rate(self.sex, self.age):
                self.demographic_features[0] = self.age
                main_part = np.abs(self.income_part['deterministic_model'].predict(self.income_part['feature_trans'].fit_transform([self.demographic_features])))
                income = abs((main_part * np.exp(gen_shock()))[0])
                hire_status = 1
            self.income_part['last_permanent_shock'] = \
                self.income_part['stochastic_param']['persistent_param'] * self.income_part['last_permanent_shock'] + self.income_part['gmm'].sample()
            self.employ_state.append(hire_status)
            self.income_state.append(income)
        else:
            income = 0    
        return income, 1
    
    def pension_benefit(self):
        def get_contribution_base(salary_bef, salary_now, ave_income, em_bef, em_now, fixed_rate=0.8, lb=0.6, ub=3):
            return max(min(fixed_rate*(salary_bef*em_bef + salary_now*(1-em_now)), ub*ave_income) , lb*ave_income)*em_now
        
        def evolve_ave(start, length, p, stochastic_param):
            res = [start]
            sto = np.random.normal(stochastic_param[0], stochastic_param[1], length)
            for i in range(1, length):
                res.append(res[-1] * p * np.exp(sto[i-1]))
            return res
        
        ave_income = self.pension_benefit_param['ave_income'][-1]
        if len(self.pension_benefit_param['con_base']) != 0:
            ave_income = self.pension_benefit_param['ave_income'][-1] * self.pension_benefit_param['ave_p'] * np.exp(np.random.normal(self.pension_benefit_param['ave_stochastic_param'][0], self.pension_benefit_param['ave_stochastic_param'][1]))
            self.pension_benefit_param['ave_income'].append(ave_income)
        
        if self.age < self.retire_age:
            self.pension_benefit_param['con_base'].append(
                get_contribution_base(
                       self.income_state[-2] if len(self.income_state) > 1 else 0, self.income_state[-1], ave_income, self.employ_state[-2] if len(self.income_state) > 1 else 0, self.employ_state[-1])
            )
            return 0, self.pension_benefit_param['con_base'][-1]
        # if ave_list not in self.pension_benefit_param:
        #     self.pension_benefit_param['ave_list'] = \
        #      evolve_ave(self.pension_benefit_param['ave_t0'], len(self.income_state), self.pension_benefit_param['ave_p'], self.pension_benefit_param['ave_stochastic_param'])
        # else:
        #     self.pension_benefit_param['ave_list'].append(\
        #         ave_list[-1] * self.pension_benefit_param['ave_p'] * np.exp(np.random.normal(self.pension_benefit_param['ave_stochastic_param'][0], self.pension_benefit_param['ave_stochastic_param'][1])))

        ave_list = self.pension_benefit_param['ave_income'] 
        # if 'con_base' not in self.pension_benefit_param:
        #     for i in range(len(self.income_state)):
        #         if i == 0:
        #             con_base = []
        #             con_base.append(get_contribution_base(
        #                 0, self.income_state[i], ave_list[i], 0, self.employ_state[i] 
        #             ))
        #             continue
        #         con_base.append(get_contribution_base(
        #             self.income_state[i-1], self.income_state[i], ave_list[i], self.employ_state[i-1], self.employ_state[i] 
        #         ))
        #     self.pension_benefit_param['con_base'] = con_base
        con_base = self.pension_benefit_param['con_base']
        
        if self.pension_benefit_param['B1_factor'] < 0:
            self.pension_benefit_param['B1_factor'] = 0.5 * (np.sum(np.array(con_base) / np.array(ave_list[:-1])) + np.sum(self.employ_state)) * 0.01
        B_1 =  self.pension_benefit_param['B1_factor'] * ave_list[-1]
        if self.pension_benefit_param['A_ret'] < 0:
            self.pension_benefit_param['A_ret'] = np.sum(np.array(con_base) * (self.pension_benefit_param['basic_pension_rate'] + self.pension_benefit_param['employer_rate']) \
                * np.array([ pow((1+self.pension_benefit_param['pension_fund_rate']), len(con_base)-i-1) for i in range(len(con_base))]))
        
        re_age = self.pension_benefit_param['retire_age'] 
        ex_age = self.pension_benefit_param['exp_life']
        rate = self.pension_benefit_param['pension_fund_rate']
        B_2 = self.pension_benefit_param['A_ret'] / ( (1-pow(1+rate, re_age - ex_age)) / (1- 1 / (1+rate)))
        return B_2 + B_1, 0

    def get_health_benfit_hs(self, age, bad_status, bef_status):
        # profile : sex,age,health_status,disease_1,disease_2,disease_3,disease_4,disease_5,disease_6,disease_7,disease_8,disease_9,disease_10,disease_11,disease_12,disease_13,disease_14
        import copy
        if age >= 75:
            return 0, 0
        now_status = np.random.choice([1, 0.74, 0], 1, p = self.health_part['trans_pro'][str(age)][str(bef_status)])[0]
        if now_status == 0:
            return 0, now_status
        
        profile = self.health_features
        poly_features = PolynomialFeatures(degree=3)
        model = self.health_part['health_cost']
        _profile = copy.deepcopy(profile)
        temp_shock = np.random.normal(self.health_part['transitory_shock'][0], self.health_part['transitory_shock'][1])
        permant_shock = np.random.normal(self.health_part['persistent_shock'][0], self.health_part['persistent_shock'][1])
        self.health_part['accumulate_shock'] = self.health_part['accumulate_shock']*self.health_part['persistent_param'] + permant_shock
        _profile[1] = profile[1] = age
        _profile[2] = bad_status
        profile[2] = now_status
        bad_cost = model.predict(poly_features.fit_transform([_profile]))
        now_cost = model.predict(poly_features.fit_transform([profile]))
        dis = bad_cost - now_cost if bad_cost > now_cost else 0

        return dis * np.exp(temp_shock + self.health_part['accumulate_shock'])*self.health_part['hs_fractor'], now_status

    def investment_decisions(self, long_r, equity_r, long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return__saving):
            long_term_bond = (1 + long_r) * self.investment_asset['long_term_bond'] + long_term_bond_saving
            equity_return = (1 + equity_r) * self.investment_asset['equity_return'] + equity_return_saving
            p_long_term_bond = (1 + long_r) * self.investment_asset['p_long_term_bond'] + p_long_term_bond_saving
            p_equity_return = (1 + equity_r) * self.investment_asset['p_equity_return'] + p_equity_return__saving
            if self.age == self.retire_age:
                p_long_term_bond *= 0.97
                p_equity_return *= 0.97
            return long_term_bond, equity_return, p_long_term_bond, p_equity_return
    
    def get_house_expenditure(self, income, house_fund_with_draw, house_fund_loan_pay, house_commerical_loan_pay, house_cost, loan_ratio, owning):
        price = self.house_param['house_price']
        own_bef = self.house_param['owning_house']
        fee = house_cost
        house_size = self.house_param['house_size']
        fund_balance = self.house_param['house_fund']
        lpr = self.house_param['lpr']
        def update_loan_term(self, bef_own, now_own):
            # print(bef_own, now_own, self.house_param['debt'])
            if bef_own == 0 and now_own == 1:
                commerical_term = self.house_param['max_commerical_loan_term']
                if self.retire_age > self.age:    
                    house_term = min(self.house_param['max_house_fund_loan_term'], self.retire_age - self.age)
                else:
                    house_term = 0
            elif bef_own == 1 and now_own == 1:
                commerical_term = self.house_param['max_commerical_loan_term'] - 1 if self.house_param['debt'] - self.house_param['house_fund_loan'] > 0 else 0
                house_term = self.house_param['max_commerical_loan_term'] - 1 if self.house_param['house_fund_loan'] > 0 else 0
            else:
                commerical_term = 0
                house_term = 0
            if self.house_param['debt'] == 0:
                self.house_param['commerical_loan_term'] = self.house_param['house_fund_loan_term'] = 0
            else:
                self.house_param['commerical_loan_term'] = commerical_term
                self.house_param['house_fund_loan_term'] = house_term
        
        if self.house_param['debt'] != 0:
            owning = 1
        # print(own_bef, owning, fund_balance, house_fund_with_draw)
        if self.age < self.retire_age:
            fund_balance = (fund_balance - house_fund_with_draw) * (1 + self.risk_free_rate) + 2 * self.house_param['house_fund_contribution_rate'] * income                                                  
        else:
            fund_balance -= house_fund_with_draw

        if own_bef == 0 and owning == 0:
            house_size = house_cost / (price * self.house_param['rent_to_price'])
            fee -= house_fund_with_draw
        elif own_bef == 0 and owning == 1:
            """"""
            house_size = house_cost / (price * (1 - loan_ratio + self.house_param['transaction_cost']))
            debt = price * house_size * loan_ratio 
            house_fund_balance_loan = self.house_param['house_loan_to_fund'] * fund_balance
            max_house_fund_loan = self.house_param['house_fund_loan_ceiling']
            if self.age < self.retire_age:
                house_fund_loan = min(house_fund_balance_loan, max_house_fund_loan, debt)
            else:
                house_fund_loan = 0
            self.house_param['debt'] = debt
            self.house_param['house_fund_loan'] = house_fund_loan
        elif own_bef == 1 and owning == 1:
            """"""
            if self.house_param['debt'] >= 100:
                fee = house_fund_loan_pay + house_commerical_loan_pay 
                debt = self.house_param['debt']
                commerical_loan = debt - self.house_param['house_fund_loan']
                house_fund_loan = self.house_param['house_fund_loan']
                # house_fund_loan = (1 + lpr - self.house_param['interest_diff']) * (house_fund_loan - house_fund_loan_pay)
                house_fund_loan = house_fund_loan - house_fund_loan_pay
                if house_fund_loan > house_fund_with_draw:
                    house_fund_loan -= house_fund_with_draw
                    house_fund_with_draw = 0
                else:
                    house_fund_with_draw -= house_fund_loan
                    house_fund_loan = 0
                house_fund_loan *= (1 + lpr - self.house_param['interest_diff'])
                commerical_loan = (1 + lpr) * (commerical_loan - house_commerical_loan_pay - house_fund_with_draw)
                self.house_param['debt'] = house_fund_loan + commerical_loan
                self.house_param['house_fund_loan'] = house_fund_loan
            else:
                fee = 0
            if self.house_param['debt'] < 100:
                self.house_param['debt'] = self.house_param['house_fund_loan'] = 0
        elif own_bef == 1 and owning == 0:
            fee = -price * house_size
        
        update_loan_term(self, own_bef, owning)
        self.house_param['owning_house'] = owning
        self.house_param['house_fund'] = fund_balance
        self.house_param['house_size'] = house_size
        return fee, house_size

    # def get_taxed_income(self, income):
    def get_new_rate(self):
        # long_r  equity_r  lpr  china_tier1  china_tier2_3
        bef = self.rate_model['rates_latest']
        bef = np.array(bef)
        if len(bef.shape) < 2:
            bef = bef.reshape((1,-1))
        model = self.rate_model['rate_model']
        ds = np.zeros((250, 5))
        for i in range(250):
            ds[i, :] = model.forecast(bef, 1)
            bef = ds[i:i+1]
        ds += 1
        # print(model.forecast(bef, 250))
        return np.cumprod(ds, axis = 0)[-1]-1, bef
    
    def get_wealth(self, long_term_bond, equity_return, p_long_term_bond, p_equity_return):
        return long_term_bond + equity_return + p_long_term_bond + p_equity_return - self.house_param['debt']
    
    def get_health_cost(self):
        def evolve_health_status(trans_pro, bef_health_status,now_age):
            if now_age >= self.exp_life or bef_health_status == 0:
                return '0', 0
            if now_age < self.retire_age:
                return '1', 1
            # print(now_age, trans_pro)
            prob = trans_pro[str(now_age)][bef_health_status]
            health_status = np.random.choice(['1', '0.74', '0'], p=prob)
            living_pro = prob[0] + prob[1]
            return health_status, living_pro
        health_features = copy.deepcopy(self.health_part['health_feature'])
        health_features[1] = self.age
        health_status, living_pro = evolve_health_status(self.health_part['trans_pro'], self.health_part['health_status'], self.age)
        model = self.health_part['health_cost']

        if health_status == '0':
            return 0, '0', 0, 0
        subsidy_factor = self.health_part['hs_fractor']
        transitory_shock = self.health_part['transitory_shock']
        persistent_param = self.health_part['persistent_param']
        persistent_shock = self.health_part['persistent_shock']
        accumulated_shock = self.health_part['accumulated_shock']
        accumulated_shock = accumulated_shock * persistent_param + np.random.normal(loc=persistent_shock[0], scale=persistent_shock[1])
        bad_health_features = copy.deepcopy(health_features)
        bad_health_features[2] = 0.74
        sto_p = accumulated_shock + np.random.normal(loc=transitory_shock[0], scale=transitory_shock[1])
        health_features[2] = float(health_status)
        poly_features = PolynomialFeatures(degree=3)
        subsidy = subsidy_factor * (abs(model.predict(poly_features.fit_transform([bad_health_features]))) - abs(model.predict(poly_features.fit_transform([health_features])))) * np.exp(sto_p)
        sim_cost = abs(model.predict(poly_features.fit_transform([health_features]))) * np.exp(sto_p)
        cost = sim_cost - subsidy
        return cost[0], health_status, accumulated_shock, living_pro

    def get_reward(self, remaining_cost, health_status, house_size, kid_num, single_child_spend, wealth, living_pro):
        risk_aversion = self.reward_param['risk_aversion']
        self_c_p = self.reward_param['opt_non_house_cost_proportion']
        c_n_i = self.reward_param['child_num_impact']
        k2 = self.reward_param['k2']
        k3 = self.reward_param['k3']
        a = self.reward_param['a']
        remaining_cost = max(remaining_cost, self.lowest_comsumption)
        # print(house_size, risk_aversion, remaining_cost, single_child_spend)
        r1 = np.power(np.power(remaining_cost, self_c_p) * np.power(house_size, 1-self_c_p), 1-risk_aversion) / (1-risk_aversion) if house_size != 0 else -1
        r2 = np.power(kid_num, c_n_i) * np.power(single_child_spend, 1-risk_aversion) / (1-risk_aversion) if single_child_spend != 0 else -1
        r3 = (1-living_pro) * np.power(wealth, 1-risk_aversion) / (1-risk_aversion) if wealth != 0 else -1
        # r = np.power(health_status, risk_aversion) * (r1+k2*r2) + k3*r3
        r = (1 - a * float(health_status)) * (r1+k2*r2) + k3*r3
        return r

    def get_real_value_of_action(self, income, kid_num, single_child_spend, owning_house, loan_ratio, house_cost, house_fund_withdraw, house_fund_loan_pay, house_commerical_loan_pay,\
                long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return_saving):
        def compute_up_low(range_arr, low_b, up_b, ori_arr):
            low = range_arr[:, 0]
            up = range_arr[:, 1]
            res_l = np.zeros_like(low)
            res_up = np.zeros_like(up)
            res = np.zeros_like(low)
            status = 1
            for i in range(len(low)-1, -1, -1):
                res_up[i] = min(up_b - low[:i].sum() - res[i+1:].sum(), up[i])
                res_l[i] = max(low_b - up[:i].sum() - res[i+1:].sum(), low[i])
                if abs(res_up[i] - res_l[i]) < 1e-5:
                    res_up[i] = res_l[i]
                res[i] = get_real_value([res_l[i], res_up[i]], ori_arr[i])
                if res_up[i] < res_l[i]:
                    status = 0
                    break
            return res, status
        
        def get_real_value(range_arr, val, factor=1):
            return (val * (range_arr[1] - range_arr[0]) + range_arr[0]) * factor
        
        def get_installment_pay(rate, time):
            return (np.power(rate, -1)) / ((1-np.power(rate, -(time+1))) / (1-np.power(rate, -1))) if time >= 0 else 0
        # balance
        own_bef = self.house_param['owning_house']
        if self.house_param['debt'] != 0:
            owning_house = 1
        ltbs_balance = self.investment_asset['long_term_bond']
        ers_balance = self.investment_asset['equity_return']
        pltbs_balance = self.investment_asset['p_long_term_bond']
        pers_balance = self.investment_asset['p_equity_return']
        hf_balance = self.house_param['house_fund']
        range_arr = 1
        res = np.zeros((10)) # loan_ratio, house_fund_loan_pay, house_commerical_loan_pay house_fund_withdraw, house_cost, single_child_spend, long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return_saving
        status = 1
        # init range
        if self.age < self.retire_age:
            cs_range = [0, income + ers_balance + ltbs_balance]
            ltbs_range = [-ltbs_balance, income + ers_balance]
            ers_range = [-ers_balance, income + ltbs_balance]
            pltbs_range = [0, income + ltbs_balance + ers_balance]
            pers_range = [0, income + ltbs_balance + ers_balance]
        else:
            cs_range = [0, income + ers_balance + ltbs_balance + pltbs_balance + pers_balance]
            ltbs_range = [-ltbs_balance, income + ers_balance + pltbs_balance + pers_balance]
            ers_range = [-ers_balance, income + ltbs_balance + pltbs_balance + pers_balance]
            pltbs_range = [-pltbs_balance, income + ltbs_balance + ers_balance + pltbs_balance + pers_balance]
            pers_range = [-pers_balance, income + ltbs_balance + ers_balance + pltbs_balance + pers_balance]
        hc_range = [0, 0]
        if own_bef == 0 and owning_house == 0:
            hc_range = [0, cs_range[1]]
            hfw_range = [-min(hc_range[1], hf_balance), 0]
            range_arr = np.array([hfw_range, hc_range, cs_range, ltbs_range, ers_range, pltbs_range, pers_range])
            ori_arr = np.array([house_fund_withdraw, house_cost, single_child_spend, long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return_saving])
            res_temp, status = compute_up_low(range_arr, -(cs_range[1] - income), income, ori_arr)
            factor = np.array([-1, 1, 1 / kid_num if kid_num != 0 else 0, \
                                        1, 1, 1, 1])
            res_temp = res_temp * factor
            res[3:] = res_temp
            # res[3:] = [get_real_value(range_arr[i, :], ori_arr[i], factor=factor[i]) for i in range(range_arr.shape[0])]
            range_arr[0, 0] = -min(res[4], -range_arr[0, 0])
            res[3] = get_real_value(range_arr[0, :], ori_arr[0], factor=factor[0])
            # print(hf_balance, res[3], res[4], range_arr[0, :], )
        elif own_bef == 0 and owning_house == 1:
            hc_range = [0, cs_range[1]]
            loan_ratio = get_real_value([0.3, 0.7], loan_ratio, 1)
            range_arr = np.array([hc_range, cs_range, ltbs_range, ers_range, pltbs_range, pers_range])
            ori_arr = np.array([house_cost, single_child_spend, long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return_saving])
            res_temp, status = compute_up_low(range_arr, -(cs_range[1] - income), income, ori_arr)
            factor = np.array([1, 1 / kid_num if kid_num != 0 else 0, \
                                        1, 1, 1, 1])
            res_temp = res_temp * factor
            res[4:] = res_temp
            res[0] = loan_ratio
        elif own_bef == 1 and owning_house == 1:
            if self.house_param['debt'] != 0:
                lpr = self.house_param['lpr']
                f_hfl = get_installment_pay(1+lpr-self.house_param['interest_diff'], self.house_param['house_fund_loan_term'])
                f_cl =  get_installment_pay(1+lpr, self.house_param['commerical_loan_term'])
                hflp_range = [f_hfl*self.house_param['house_fund_loan'], min(self.house_param['house_fund_loan'], income + ers_balance + ltbs_balance + hf_balance)]
                hclp_range = [f_cl*(self.house_param['debt']-self.house_param['house_fund_loan']), min(self.house_param['debt'] - self.house_param['house_fund_loan'], income + ers_balance + ltbs_balance + hf_balance)]
                hfw_range = [-min(self.house_param['debt'], hf_balance), 0]
                hc_range = [0, 0]
                range_arr = np.array([hflp_range, hclp_range, hfw_range, hc_range, cs_range, ltbs_range, ers_range, pltbs_range, pers_range])
                ori_arr = np.array([house_fund_loan_pay, house_commerical_loan_pay, house_fund_withdraw, house_cost, single_child_spend, long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return_saving])
                res_temp, status = compute_up_low(range_arr, -(cs_range[1] - income), income, ori_arr)
                factor = np.array([1, 1, -1, 1, 1 / kid_num if kid_num != 0 else 0, \
                                        1, 1, 1, 1])
                res_temp = res_temp * factor
                res[1:] = res_temp
                # print(self.age, f_hfl, hf_balance, hflp_range, hclp_range)
            else:
                range_arr = np.array([cs_range, ltbs_range, ers_range, pltbs_range, pers_range])
                ori_arr = np.array([single_child_spend, long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return_saving])
                res_temp, status = compute_up_low(range_arr, -(cs_range[1] - income), income, ori_arr)
                factor = np.array([1. / kid_num if kid_num != 0 else 0, \
                                        1., 1., 1., 1.])
                res_temp = res_temp * factor
                res[5:] =  res_temp
        # print(res)
        # print(own_bef, owning_house, hc_range)
        # single_child_spend [0, ]
        return res, status

    def step(self, state, action: list):
        # print(action)
        # state age, health_status, health_care_cost, living_pro, kid_num, income, contribution_base, pension_benfit, house_fund_balance, asset_liquid, asset_pension, debt_commerical, debt_house_fund, house_price 
        age, health_status, health_care_cost, living_pro, kid_num, income, con_base, own_bef, house_fund_balance, long_term_bond, equity_return, p_long_term_bond, p_equity_return, debt_commerical, debt_house_fund, house_price\
            = state
        single_child_spend, house_cost, loan_ratio, house_fund_withdraw, house_fund_loan_pay, house_commerical_loan_pay, long_term_bond_saving, \
            equity_return_saving, p_long_term_bond_saving, p_equity_return_saving \
            = action[1]
        owning_house = action[0]
        # print(self.age, income, health_care_cost, income-health_care_cost)
        if income < health_care_cost:
            income = 0
            health_care_cost -= income
            for asset in self.investment_asset:
                if self.investment_asset[asset] >= health_care_cost:
                    self.investment_asset[asset] -= health_care_cost
                    health_care_cost = 0
                    break
                health_care_cost -= self.investment_asset[asset]
                self.investment_asset[asset] = 0
            if health_care_cost != 0:
                return state, -3, True
        else:
            income -= health_care_cost
        real_actions, status = self.get_real_value_of_action(income=income, kid_num=kid_num, single_child_spend=single_child_spend, owning_house=owning_house, loan_ratio=loan_ratio, house_cost=house_cost,
                                          house_fund_withdraw=house_fund_withdraw, house_fund_loan_pay=house_fund_loan_pay, house_commerical_loan_pay=house_commerical_loan_pay,
                                          long_term_bond_saving=long_term_bond_saving, equity_return_saving=equity_return_saving, p_equity_return_saving=p_equity_return_saving,
                                          p_long_term_bond_saving = p_long_term_bond_saving)
        if status == 0:
            return state, -3, True # broken
        
        loan_ratio, house_fund_loan_pay, house_commerical_loan_pay, house_fund_withdraw, house_cost, single_child_spend, long_term_bond_saving, equity_return_saving, p_long_term_bond_saving, p_equity_return_saving\
          = real_actions
        
    
        # house
        house_fee, house_size = self.get_house_expenditure(con_base, house_fund_withdraw, house_fund_loan_pay, house_commerical_loan_pay, \
                       house_cost, loan_ratio, owning_house)
        # wealth
        wealth = self.get_wealth(long_term_bond, equity_return, p_long_term_bond, p_equity_return) + self.house_param['owning_house'] * self.house_param['house_price'] * self.house_param['house_size']
        # non-consumption
        if self.age < self.retire_age:
            non_consumption = income - self.pension_benefit_param['basic_pension_rate'] * con_base - single_child_spend * kid_num - house_fee -\
                (long_term_bond_saving + equity_return_saving + p_long_term_bond_saving + p_equity_return_saving) - health_care_cost
        else:
            non_consumption = income + house_fund_withdraw - (1 - 0.03) * (p_long_term_bond_saving + p_equity_return_saving) - single_child_spend * kid_num - house_fee - health_care_cost
        reward  = self.get_reward(non_consumption, health_status, house_size, kid_num, single_child_spend, wealth, living_pro)

        # generate next state
        # compute 
        self.age += 1
        # print(self.age)
        kid_num = self.child_num[self.age]
        # compute income
        income_next = self.evlove_income()[0]
        pension_bef, con_base = self.pension_benefit()
        income_next += pension_bef
        # health 
        health_care_cost = 0
        health_status = 1
        living_pro = 1
        if self.age >= self.retire_age:
            health_care_cost, health_status, accumulated_shock, living_pro = self.get_health_cost()
            self.health_part['accumulated_shock'] = accumulated_shock
            self.health_part['health_status'] = health_status
        else:
            health_care_cost = self.health_part['hs_fractor'] * self.health_part['g']
        
        # rates
        
        rates, rates_latest = self.get_new_rate()
        long_r, equity_r, lpr, china_tier1, china_tier2_3 = rates
        self.rate_model['rates'] = rates
        self.rates_latest = rates_latest

        # investment compute
        long_term_bond, equity_return, p_long_term_bond, p_equity_return = self.investment_decisions(long_r, equity_r, long_term_bond_saving, \
                equity_return_saving, p_long_term_bond_saving, p_equity_return_saving)
        self.investment_asset['long_term_bond'] = long_term_bond
        self.investment_asset['equity_return'] = equity_return
        self.investment_asset['p_long_term_bond'] = p_long_term_bond
        self.investment_asset['p_equity_return'] = p_equity_return
        # tex a1 * tex^a2,  a1 = 0.15, a2 = 0.98
        exp_tex = 12000 * kid_num
        if self.house_param['owning_house'] == 0:
            exp_tex += 12000
        if self.house_param['owning_house'] == 1 and self.house_param['debt'] > 0:
            exp_tex += 12000
        if income >= 60000 and self.age < self.retire_age:
            income = income_next + 0.15 * np.power(exp_tex, 0.98) 
        self.house_param['house_price'] = self.house_param['house_price'] * (1+china_tier1)

        new_state = [
            self.age, float(health_status), health_care_cost, living_pro, kid_num, income, con_base,\
            self.house_param['owning_house'], self.house_param['house_fund'], long_term_bond, equity_return, p_long_term_bond, p_equity_return,\
            self.house_param['debt'] - self.house_param['house_fund_loan'], self.house_param['house_fund_loan'], self.house_param['house_price']
        ]
        
        done = False
        if health_status == '0':
            done = True
        return new_state, reward, done