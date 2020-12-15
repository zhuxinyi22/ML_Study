from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import numpy as np
iris = load_iris()

X = iris.data
scaler = MinMaxScaler()

scaler.fit(X)

X_std = scaler.transform(X)

addon = np.ones((50,1))

label = np.zeros((50,1))

label = np.r_[label,addon]

addon = addon * 2

label = np.r_[label,addon]

X_std = np.c_[label,X_std]

print(X_std)

np.savetxt('iris.csv',X_std,fmt='%.4f',delimiter=',')
