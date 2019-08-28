import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pickle
import time
from sklearn.model_selection import train_test_split
import os
import random
import copy
import pandas as pd

def MAPE_error(y_test, y_pred):
    return np.mean(abs(y_test - y_pred)/y_test*100)
def MSE_error(y_test, y_pred):
    return np.mean((y_test - y_pred)**2)
def MSE_relative_error(y_test, y_pred):
    return np.mean(((y_test - y_pred)**2)/y_pred)

os.chdir("YOUR PATH HERE")
train = pd.read_csv("trip_travel_2015_modified.csv")
y = train["TripTotalTime"].values
X = train.drop(["TripTotalTime"], axis = 1).values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(time.strftime('%H:%M%p %Z on %b %d, %Y'))

classifiers =  [
                ("SVR1", SVR()),
                ("SVR2", SVR(kernel='linear')),
                ("SVR3", SVR(kernel='poly')),
                ("SVR4", SVR(kernel='sigmoid')),
                ("RF1", RandomForestRegressor()),
                ("RF11", RandomForestRegressor(max_depth=50, min_samples_split=10)),
                ("RF2", RandomForestRegressor(n_estimators=100, max_depth=4)),
                ("RF2", RandomForestRegressor(n_estimators=1000, max_depth=50, min_samples_split=10)),
                ("RF3", RandomForestRegressor(n_estimators=1000, max_depth=50, min_samples_split=4)),
                # ("RF4", RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=10)),
                # ("RF5", RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=4))]
    ]

n_runs = 1
clf_d = {j[0]: [] for j in classifiers}

time_start = time.clock()
print(time_start)

for name, clf in classifiers:
    print('Training model {}'.format(name))

    res_b = {'MAPE': 0, 'MSE': 0, 'MSER': 0}

    for iterator in range(n_runs):
        print('Run # {}'.format(iterator))
        random_st = np.random.randint(134, 789)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15, random_state=random_st)

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        res_b['MAPE'] += MAPE_error(Y_test, Y_pred)
        res_b['MSE'] += MSE_error(Y_test, Y_pred)
        res_b['MSER'] += MSE_relative_error(Y_test, Y_pred)

    print("Model: ", name)
    for k in res_b.keys():
        print(k, (res_b[k]/float(n_runs)))
    print("# -"*20)

print(time.strftime('%H:%M%p %Z on %b %d, %Y'))

time_elapsed = (time.clock() - time_start)
print("Time to run: ", time_elapsed)
