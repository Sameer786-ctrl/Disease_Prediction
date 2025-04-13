# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 09:56:30 2025

@author: sam77
"""
#Training data frame 1 imported and names
import pandas as pd
df1=pd.read_csv("C:/Btech-studies/SEM4/Hons_AIML/Training.csv")
df1 = df1.drop(columns=['Unnamed: 133'])
print(df1)
x_train=df1.drop('prognosis',axis=1)
x_train
y_train=df1['prognosis']
y_train


# Testing data frame 1 imported and names
df2=pd.read_csv("C:/Btech-studies/SEM4/Hons_AIML/Testing.csv")
print(df2.isna().sum().sum())
print(df2)
x_test=df2.drop('prognosis',axis=1)
x_test
y_test=df2['prognosis']
y_test

#training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

#testing the model
y_pred=model.predict(x_test)

#tAccuracy test
from sklearn.metrics import accuracy_score
accuracy=accuracy_score












