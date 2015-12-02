# In the name of God
__author__ = 'esadeqia'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

trainDataFileAddress=".//data//test.csv"
x=pd.read_csv(trainDataFileAddress, engine='python')

columns=x.columns.values
print(columns)
exit(0)

trainData=x.values



print(type(x))
print(type(trainData))
print(trainData[:4])
print(columns)
print(type(columns))
print(np.array([2,3,4]))

# y=x.dropna()
# y=x.fillna(x.mean())
# print(y)
print("hello")
tedad=x.isnull().sum()
for i in x.isnull().values:
    for j in i:
        if j:
            print("there is")
print("end")
print(tedad)
