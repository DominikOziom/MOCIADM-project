# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from collections import Counter

from utils import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from my_method_class import SyntheticOverSampler

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt




#data = pd.read_csv('data/yeast1.csv')
data = pd.read_csv('data/yeast6.dat')
X, y, classes = prepare_data(data)

cnt = Counter()
res_cnt = Counter()

algorithms = []
names = []

# OVER SAMPLING
sos = SyntheticOverSampler()
algorithms.append(sos)
names.append("GSO")

sm = SMOTE()
algorithms.append(sm)
names.append("SMOTE")

adas = ADASYN()
algorithms.append(adas)
names.append("ADASYN")

# COMBINE
sm_t = SMOTETomek()
algorithms.append(sm_t)
names.append("SMOTETomek")


fpr = [[], [], []]
tpr = [[], [], []]

knn_neigh =  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  #range of neighbors.  default 5
rf_estim =   [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]   #number of trees.     default 10
mlp_l_size = [10, 30, 40, 50, 60, 70, 80, 90, 100, 110] # hidden layer size. default 100

i = 0
for alg in algorithms:
    best_score = [0, 0, 0]
    param = [0, 0, 0]
    fpr[0].append(0)
    tpr[0].append(0)
    tpr[1].append(0)
    fpr[1].append(0)
    fpr[2].append(0)
    tpr[2].append(0)
    res_X, res_y = alg.fit_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(res_X, res_y, test_size=0.70)
    
    for g in range(len(rf_estim)):
        print("alg {}. {} out of {}".format(i+1, g+1, len(rf_estim)))

        # KNN
        kNN = KNeighborsClassifier(n_neighbors=knn_neigh[g])
        kNN.fit(X_train, y_train)
        y_pred = kNN.predict(X_test)
        y_pred_m = list(map(int, y_pred))
        y_test_m = list(map(int, y_test))
        fpr1, tpr1, thresholds = roc_curve(y_test_m, y_pred_m, pos_label=1)
        fpr[0].append(fpr1[1])
        tpr[0].append(tpr1[1])
        score1 = recall_score(y_pred, y_test, average='micro')
        score2 = accuracy_score(y_pred, y_test)
        val = score1*score2
        if val>best_score[0]:
            best_score[0] = val
            param[0] = knn_neigh[g]
        # RF
        rf = RandomForestClassifier(n_estimators=rf_estim[g])
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_pred_m = list(map(int, y_pred))
        y_test_m = list(map(int, y_test))
        fpr1, tpr1, thresholds = roc_curve(y_test_m, y_pred_m, pos_label=1)
        fpr[1].append(fpr1[1])
        tpr[1].append(tpr1[1])
        score1 = recall_score(y_pred, y_test, average='micro')
        score2 = accuracy_score(y_pred, y_test)
        val = score1*score2
        if val>best_score[1]:
            best_score[1] = val
            param[1] = rf_estim[g]
        # MLP
        n = mlp_l_size[g]
        mlp = MLPClassifier(hidden_layer_sizes=(n,n,n), max_iter=400)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        y_pred_m = list(map(int, y_pred))
        y_test_m = list(map(int, y_test))
        fpr1, tpr1, thresholds = roc_curve(y_test_m, y_pred_m, pos_label=1)
        fpr[2].append(fpr1[1])
        tpr[2].append(tpr1[1])
        score1 = recall_score(y_pred, y_test, average='micro')
        score2 = accuracy_score(y_pred, y_test)
        val = score1*score2
        if val>best_score[2]:
            best_score[2] = val
            param[2] = mlp_l_size[g]
        

    fpr[0].append(1)
    tpr[0].append(1)
    tpr[1].append(1)
    fpr[1].append(1)
    fpr[2].append(1)
    tpr[2].append(1)
    
    aucs = []
    aucs.append(auc(fpr[0], tpr[0], reorder=True))
    aucs.append(auc(fpr[1], tpr[1], reorder=True))
    aucs.append(auc(fpr[2], tpr[2], reorder=True))
    
    fpr[0].sort()
    fpr[1].sort()
    fpr[2].sort()
    tpr[0].sort()
    tpr[1].sort()
    tpr[2].sort()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr[0], tpr[0], 'r-^', label="kNN")
    plt.plot(fpr[1], tpr[1], 'b-o', label="RF")
    plt.plot(fpr[2], tpr[2], 'g-s', label="MLP")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for ' + names[i])
    plt.legend()
    plt.grid(True)
    plt.show()


    print("\t\t Algorithm: {}".format(names[i].upper()))
    print("\t      kNN    RandomForest     MLP")
    print("  score: {0:10.4f}  {1:10.4f}  {2:10.4f}".format(best_score[0], best_score[1], best_score[2]))
    print("  param: {0:10.0f}  {1:10.0f}  {2:10.0f}".format(param[0], param[1], param[2]))
    print("    AUC: {0:10.4f}  {1:10.4f}  {2:10.4f}\n".format(aucs[0], aucs[1], aucs[2]))
    
    fpr[0].clear()
    fpr[1].clear()
    fpr[2].clear()
    tpr[0].clear()
    tpr[1].clear()
    tpr[2].clear()
    aucs.clear()
    i+=1



