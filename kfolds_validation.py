from __future__ import print_function, division
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class ModelFactory(object):
    def __init__(self, model_type):
        self.model_type = model_type
    
    def factory(self):
        if self.model_type is NEURAL_NET_MODEL:
            return NeuralNetModel()
        elif self.model_type is SVM_MODEL:
            return SVMModel()
        elif self.model_type is RFC_MODEL:
            return RFCModel()
        
class ModelBase(object):
    def __init__(self):
        pass
    
    def build_model(self):
        raise Exception('Not Implemented')
    
class NeuralNetModel(ModelBase):
    def __init__(self):
        pass
    
    def build_model(self):
        return MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2),
                        random_state=None) 

class SVMModel(ModelBase):
    def __init__(self):
        pass
    
    def build_model(self):
        return svm.SVC(random_state=None)

class RFCModel(ModelBase):
    def __init__(self):
        pass
    
    def build_model(self):
        return RandomForestClassifier()
    
def run_cross_validation_models(df, kfold_value, model_type):
    kf = KFold(n_splits = kfold_value, random_state=None, shuffle=False)
    y_cols = 1
    x_cols = list(set(df.columns) - {0, 1})
    k_val = 5
    
    y = df[y_cols]
    X = df[x_cols]
    score_list = []
    
    kf = KFold(n_splits = k_val, random_state=None, shuffle=False)
    clf = ModelFactory(model_type).factory().build_model()
    
    for train_index, test_index in kf.split(X):
        clf.fit(X.iloc[train_index].values, y.iloc[train_index].values)
        predict = clf.predict(X.iloc[test_index].values)
        score = accuracy_score(y.iloc[test_index].values, predict)
        score_list.append(score)
    
    score_arr = np.array(score_list)
    mean_value, std_value = np.mean(score_arr), np.std(score_arr)
    
    print('Mean Value {} STD Value {} : from KFold with model {}'.format(mean_value, std_value, model_type))
    cv_scores = cross_val_score(clf, X.values, y.values, cv=5)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    print('Mean Value {} STD Value {}: from Cross Val Score with model {}'.format(cv_mean, cv_std, model_type))
    
    if abs(cv_mean - mean_value) <= min(cv_std, std_value):
        print('Cross Validation Means within the min of the STDs with model {}'.format(model_type))
    else:
        print('Cross Validation Means outside the min of the STDs with model {}'.format(model_type))
    

NEURAL_NET_MODEL = 0
SVM_MODEL = 1
RFC_MODEL = 2
def main():
    df = pd.read_csv('C:\\Users\\mitch\\Desktop\\Masters\\DataMining2'
                   +  '\\KFoldsValidation\\WisconsinBreastCancer\\wdbc_data_abclean.csv',
                   header=None)
    kfold_value = 5
    for model_type in [NEURAL_NET_MODEL, SVM_MODEL, RFC_MODEL]:        
        run_cross_validation_models(df, kfold_value, model_type)

if __name__ == '__main__':
    main()