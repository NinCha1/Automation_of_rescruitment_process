import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from  imblearn.under_sampling import RandomUnderSampler 
from collections import Counter

def undersampling(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_undersampling, y_undersampling = rus.fit_resample(X, y)
    return X_undersampling, y_undersampling

def oversampling(X, y):
    sm = SMOTE(random_state=42)
    X_oversampling, y_oversampling = sm.fit_resample(X, y)

    return X_oversampling, y_oversampling



