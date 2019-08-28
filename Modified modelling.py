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
import matplotlib.pyplot as plt

def MAPE_error(y_test, y_pred):
    return np.mean(abs(y_test - y_pred)/y_test*100)
def MSE_error(y_test, y_pred):
    return np.mean((y_test - y_pred)**2)
def MSE_relative_error(y_test, y_pred):
    return np.mean(((y_test - y_pred)**2)/y_pred)

os.chdir("YOUR PATH HERE")
train = pd.read_csv("trip_travel_2015.csv") #Initial

train_1 = train # num stages=1, mode unchanged
train_1['NumStages'] = 1

train_2 = pd.read_csv("trip_travel_2015_mod.csv") # modified mode, num stages unchanged
train_2['NumStages'] = train['NumStages']

train_3 = pd.read_csv("trip_travel_2015_mod.csv") # modified mode, num stages included
train_3['NumStages'] = 1

print(train.shape, train_1.shape, train_2.shape, train_3.shape)

y = train["TripTotalTime"].values
y_1 = train_1["TripTotalTime"].values
y_2 = train_2["TripTotalTime"].values
y_3 = train_3["TripTotalTime"].values

X = train.drop(["TripTotalTime"], axis = 1).values
X_1 = train_1.drop(["TripTotalTime"], axis = 1).values
X_2 = train_2.drop(["TripTotalTime"], axis = 1).values
X_3 = train_3.drop(["TripTotalTime"], axis = 1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.20, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.20, random_state=42)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.20, random_state=42)

print(time.strftime('%H:%M%p %Z on %b %d, %Y'))

time_start = time.clock()
print(time_start)

res_b = {'MAPE': 0, 'MSE': 0, 'MSER': 0}

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
print("PREDICTIONS STARTED")
Y_pred = clf.predict(X_test)
Y_pred_1 = clf.predict(X_test_1)
Y_pred_2 = clf.predict(X_test_2)
Y_pred_3 = clf.predict(X_test_3)

res = y_test - Y_pred
res_1 = y_test_1 - Y_pred_1
res_2 = y_test_2 - Y_pred_2
res_3 = y_test_3 - Y_pred_3

print(MSE_relative_error(y_test, Y_pred))
print(MSE_relative_error(y_test_1, Y_pred_1))
print(MSE_relative_error(y_test_2, Y_pred_3))
print(MSE_relative_error(y_test_3, Y_pred_2))

def plot_results(res, name):

    plt.plot(res, marker='.')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel("Respondent")
    plt.ylabel("True - Predicted")
    plt.savefig(name)

def plot_results_scatter(y, y1, name):

    plt.scatter(y, y1)
    plt.plot(y, y, 'r')
    plt.xlabel("True time")
    plt.ylabel("Predicted time")
    plt.show()
    plt.savefig(name)
    print("Figure saved!")

# plot_results(res[2000:2100], "initial.png")
# plot_results(res_1[2000:2100], "first_mode.png")
# plot_results(res_2[2000:2100], "second_mode.png")
# plot_results(res_3[2000:2100], "third_mode.png")

# plot_results_scatter(y_test[2000:2100], Y_pred[2000:2100], "initial_scatter.png")
# plot_results_scatter(y_test_1[2000:2100], Y_pred_1[2000:2100], "first_scatter")
# plot_results_scatter(y_test_2[2000:2100], Y_pred_2[2000:2100], "first_scatter")
plot_results_scatter(y_test_3[2000:2100], Y_pred_3[2000:2100], "first_scatter")

print(time.strftime('%H:%M%p %Z on %b %d, %Y'))

time_elapsed = (time.clock() - time_start)
print("Time to run: ", time_elapsed)
#print("Accuracy: ", acc_matr)