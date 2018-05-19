from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn import svm

NEURAL_NET_MODEL = 0
SVM_MODEL = 1
RFC_MODEL = 2
ETC_MODEL = 3 
GBC_MODEL = 4
DT_MODEL = 5

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
        elif self.model_type is ETC_MODEL:
            return ETCModel()
        elif self.model_type is GBC_MODEL:
            return GBCModel()
        elif self.model_type is DT_MODEL:
            return DTModel()
        else:
            raise Exception('{} Model Type not Implemented'.format(self.model_type))

        
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

class ETCModel(ModelBase):
    def __init__(self):
        pass
    
    def build_model(self):
        return ExtraTreesClassifier()

class GBCModel(ModelBase):
    def __init__(self):
        pass
    
    def build_model(self):
        return GradientBoostingClassifier()

class DTModel(ModelBase):
    def __init__(self):
        pass
    
    def build_model(self):
        return DecisionTreeClassifier()