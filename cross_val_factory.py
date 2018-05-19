from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit#, StratifiedKFold, LeavePOut

class CrossValFactory(object):
    def __init__(self, cross_val):
        self.cross_val = cross_val
    
    def factory(self, k_val):
        if self.cross_val is KF_VAL:
            return KFold(n_splits = k_val, random_state=None, shuffle=False)
        elif self.cross_val is LOO_VAL:
            return LeaveOneOut()
        #elif self.cross_val is LPO_VAL:
        #    return LeavePOut(p=k_val)
        elif self.cross_val is SS_VAL:
            return ShuffleSplit(n_splits=k_val)
        #elif self.cross_val is SKF_VAL:
        #    return StratifiedKFold(n_splits=k_val)
        else:
            raise Exception('Cross Validation Type {} not Implemented'.format(self.cross_val))



KF_VAL = 0 
LOO_VAL = 1
#LPO_VAL = 2
SS_VAL = 3
#SKF_VAL = 4
