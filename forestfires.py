# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:55:22 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\\chetan\\assignment\\16.SVM\\forestfires.csv")

df.info()
df.shape

x = df.iloc[:,3:30]
y = df.iloc[:,30:]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=200)
x_train.shape
x_test.shape
y_test.shape
y_train.shape


print("hellow world")

df.columns
df.hist("FFMC")

model_1 = SVC(kernel='linear',C =5)
help(SVC)
model_1.fit(x_train,y_train)
model_1.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,model_1.predict(x_test))
model_1.score(x_test,y_test)

model_2 = SVC(kernel='rbf',C =5,gamma= 'scale')
model_2.fit(x_train,y_train)
model_2.score(x_train,y_train)
model_2.score(x_test,y_test)


model_3 = SVC(kernel='sigmoid',C =5,gamma= 'scale',coef0=0.5)
model_3.fit(x_train,y_train)
model_3.score(x_train,y_train)
model_3.score(x_test,y_test)

from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,50,100,500],'gamma':[1,0.5,0.1,0.01,0.001]}
grid= GridSearchCV(SVC(),param_grid, verbose=3, n_jobs=-1)
grid.fit(x_train,y_train)
grid.best_params_


best_model = SVC(kernel='rbf',C =50,gamma= 0.001)
best_model.fit(x_train,y_train)
best_model.score(x_train,y_train)
best_model.score(x_test,y_test)








