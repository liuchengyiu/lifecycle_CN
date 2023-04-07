from datasets import DataSet
from environment import env
from functools import wraps



enviorment = env(DataSet(), 20, 1, [20.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,])
print(enviorment.evlove_income())
enviorment.age += 1

print(enviorment.evlove_income())
enviorment.age += 1
print(enviorment.evlove_income())
enviorment.age += 1
print(enviorment.evlove_income())
enviorment.age += 1
print(enviorment.evlove_income())
enviorment.pension_benefit()
# print(enviorment.income_state)
# print(enviorment.employ_state)