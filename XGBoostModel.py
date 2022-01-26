#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:39:06 2021

@author: jaimegde
"""


#Data wrangling
import numpy as np
import pandas as pd
import datetime as dt

#Data visualization
from plotnine import *
import matplotlib.pyplot as plt
plt.style.use("seaborn")

#XGBoost model
from xgboost import XGBClassifier

#SKLearn
#Metrics
import sklearn.metrics as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
#Models
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.model_selection import learning_curve, ShuffleSplit, validation_curve, KFold
from sklearn.model_selection import train_test_split


#%%
#LOAD DATASET : MASTER_DF2

dataset = pd.read_csv('master_df.csv')
dataset['fecha'] = pd.to_datetime(dataset.fecha)

#Define Target (y)
y = dataset.loc[:,['fecha','target_75']]
y.set_index('fecha', inplace=True)

#Define Features (X)
X = dataset.loc[:,columns_model]
X.set_index('fecha', inplace=True)


columns_model = ['fecha',
       'avg_cases_14d', 'avg_cases_54d',
       'daily_vac_14d', 'viajeros_14d',
       'viajeros_15d', 'num_covid_cases_16d', 'num_covid_cases_17d', 'num_covid_cases_18d',
       'num_covid_cases_19d', 'num_covid_cases_20d', 'num_covid_cases_30d',
       'num_covid_cases_3m', 'festivo', 'weekend','Fases14_d', 'Telework mandatory 14_d', 'Telework recomended14_d',
       'Education14_d', 'RetailStores14_d', 'Rest_terrazas14_d',
       'Rest_Club14_d', 'Mobility oustide municipio14_d', 'schoolday14_d',
       'weekend14_d', 'Fases21_d', 'Telework mandatory 21_d',
       'Telework recomended21_d', 'Education21_d', 'RetailStores21_d',
       'Rest_terrazas21_d', 'Rest_Club21_d', 'Mobility oustide municipio21_d',
       'schoolday21_d', 'weekend21_d', 'Fases30_d', 'Telework mandatory 30_d',
       'Telework recomended30_d', 'Education30_d', 'RetailStores30_d',
       'Rest_terrazas30_d', 'Rest_Club30_d', 'Mobility oustide municipio30_d',
       'schoolday30_d', 'weekend30_d']


#1. ======> GRID SEARCH FOR BEST PARAMS:

skf = StratifiedKFold(n_splits=10)
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=4,shuffle=True, stratify=y)

xgb = XGBClassifier()
gs = GridSearchCV(xgb, cv=skf, param_grid=params, return_train_score=True)
gs.fit(X, y)

gs.best_score_
best_model = gs.best_estimator_
best_params = gs.best_params_

a = gs.cv_results_

best_model.score(X_test, y_test)

#%%
#2. =========> RUN THE MODEL WITH THE BEST PARAMS STORED IN BEST_PARAMS


#Define Target (y)
y = dataset.loc[:,['fecha','target_75']]
y.set_index('fecha', inplace=True)

#Define Features (X)
X = dataset.loc[:,columns_model]
X.set_index('fecha', inplace=True)

#Split test_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, shuffle=True)

#Define model
model = XGBClassifier(gamma=1.5, max_depth=4, min_child_weight=1, subsample=1.0,
                      colsample_bytree=0.8)
#Fit model
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

#Accuracy
accuracy = accuracy_score(y_pred, y_test)

#Confusion matrix and plot
sm.confusion_matrix(y_pred, y_test)

fig, ax = plt.subplots(constrained_layout=False)
plot_confusion_matrix(model, X_test, y_test, ax=ax)
ax.grid(b=False)

#Feature importance
features = pd.DataFrame(data = model.feature_importances_, index = X_train.columns, columns=['importance'])
features = features.sort_values(by=['importance'],ascending=False)


##### WE SELECT THE RELEVANT VARIABLES ######

important_features = ['fecha', 'daily_vac_14d', 'Fases30_d', 'num_covid_cases_30d', 'viajeros_14d',
       'viajeros_15d', 'num_covid_cases_18d', 'Fases14_d', 'festivo',
       'avg_cases_14d', 'num_covid_cases_3m', 'num_covid_cases_20d',
       'avg_cases_54d', 'num_covid_cases_19d', 'num_covid_cases_17d',
       'weekend21_d']


#%%
#3. ====> WE RUN THE MODEL AGAIN ONLY WITH THE RELEVANT VARIABLES:

    
#Define Target (y)
y = dataset.loc[:,['fecha','target_75']]
y.set_index('fecha', inplace=True)

#Define Features (X)
X = dataset.loc[:,important_features] #WE SELECT THE IMPORTANT FEATURES AS X
X.set_index('fecha', inplace=True)

#Split test_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, shuffle=True)

#Define model
model = XGBClassifier(gamma=1.5, max_depth=4, min_child_weight=1, subsample=1.0,
                      colsample_bytree=0.8)
#Fit model
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

#Accuracy
accuracy = accuracy_score(y_pred, y_test)

#Confusion matrix and plot
sm.confusion_matrix(y_pred, y_test)

fig, ax = plt.subplots(constrained_layout=False)
plot_confusion_matrix(model, X_test, y_test, ax=ax)
ax.grid(b=False)

#Feature importance
features = pd.DataFrame(data = model.feature_importances_, index = X_train.columns, columns=['importance'])
features = features.sort_values(by=['importance'],ascending=False)

#%%
#4. CROSS VALIDATION

#LOAD DATA

y = dataset.loc[:,['fecha','target_75']]
y.set_index('fecha', inplace=True)

#Define Features (X)
X = dataset.loc[:,important_features] #WE SELECT THE IMPORTANT FEATURES AS X
X.set_index('fecha', inplace=True)


# CV model
model_cv = XGBClassifier(gamma=1.5, max_depth=4, min_child_weight=1, subsample=1.0,
                      colsample_bytree=0.8)
kfold = KFold(n_splits=10, random_state=4, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# AND NOW. THATS OUR BEAUTIFUL MODEL! <3 <3 <3



# define model
#model = XGBClassifier(scale_pos_weight=99)

#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
#print('Mean ROC AUC: %.5f' % mean(scores))
        

