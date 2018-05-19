from __future__ import print_function, division
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import model_factory
import cross_val_factory


#models
NEURAL_NET_MODEL = 0
SVM_MODEL = 1
RFC_MODEL = 2
ETC_MODEL = 3 
GBC_MODEL = 4
DT_MODEL = 5

#Cross Val Methods
KF_VAL = 0 
LOO_VAL = 1
#LPO_VAL = 2
SS_VAL = 3
#SKF_VAL = 4

def run_cross_validation_models(df, kfold_value, model_type, cross_val, iterable=None):
    #kf = KFold(n_splits = kfold_value, random_state=None, shuffle=False)
    y_cols = 1
    x_cols = list(set(df.columns) - {0, 1})
    k_val = 5
    
    y = df[y_cols]
    X = df[x_cols]
    score_list = []
    cross_val_method = cross_val_factory.CrossValFactory(cross_val).factory(k_val)
    clf = model_factory.ModelFactory(model_type).factory().build_model()
    
    for train_index, test_index in cross_val_method.split(X):
        clf.fit(X.iloc[train_index].values, y.iloc[train_index].values)
        predict = clf.predict(X.iloc[test_index].values)
        score = accuracy_score(y.iloc[test_index].values, predict)
        score_list.append(score)
    
    score_arr = np.array(score_list)
    mean_value, std_value = np.mean(score_arr), np.std(score_arr)
    
    print('Mean Value {} STD Value {} : from KFold with model {} using cross validation type {}'.format(mean_value, std_value, model_type, cross_val))
    cv_scores = cross_val_score(clf, X.values, y.values, cv=5)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    print('Mean Value {} STD Value {}: from Cross Val Score with model {} using cross validation type {}'.format(cv_mean, cv_std, model_type, cross_val))
    
    if abs(cv_mean - mean_value) <= min(cv_std, std_value):
        print('Cross Validation Means within the min of the STDs with model {} with cross validation type {}'.format(model_type, cross_val))
        return 1
    else:
        print('Cross Validation Means outside the min of the STDs with model {} with cross validation type {}'.format(model_type, cross_val))
        return 0

def main():
    df = pd.read_csv('C:\\Users\\mitch\\Desktop\\Masters\\DataMining2'
                   +  '\\KFoldsValidation\\WisconsinBreastCancer\\wdbc_data_abclean.csv',
                   header=None)
    kfold_value = 5
    multiple_runs = False
    multiple_run_count = 100
    for model_type in [NEURAL_NET_MODEL, SVM_MODEL, RFC_MODEL, ETC_MODEL, GBC_MODEL, DT_MODEL]:
        for cross_val in [KF_VAL, LOO_VAL, SS_VAL]:
            print('Using Model {} with Cross Val Type {}'.format(model_type, cross_val))
            if multiple_runs:
                pool = Pool()
                partial_cross_val = partial(run_cross_validation_models, df, kfold_value, model_type, cross_val)
                results = pool.map(partial_cross_val, [i for i in range(multiple_run_count)])
                print('{} Percentage of Model Type {} had means fall within the min of the STDs of the models using cross validation type {}'.format(np.mean(results)*100, model_type, cross_val))
            else:
                run_cross_validation_models(df, kfold_value, model_type, cross_val)

if __name__ == '__main__':
    main()