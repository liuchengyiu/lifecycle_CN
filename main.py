from datasets import DataSet
from environment import env



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