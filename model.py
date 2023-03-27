import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("salary.csv")
dataset.isna().sum()
dataset['experience'].fillna(0,inplace=True)
dataset
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)
dataset.isnull().sum()
dataset['experience'] = dataset['experience'].map({'five':5,'two':2,'seven':7,'three':3,'ten':10,'eleven':11,0:0})
dataset.info()

x = dataset.iloc[:,:-1]
y = dataset['salary']
y

from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.model_selection import train_test_split
## model building
regressor = model.fit(x,y)
### saving model 
pickle.dump(regressor,open('model.pkl','wb'))

model_test = pickle.load(open('model.pkl','rb'))
print(model_test.predict([[2,8,9]]))
print(model_test.predict([[5,3,5]]))