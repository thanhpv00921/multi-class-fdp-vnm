import numpy as np
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
import pickle

# load trained model.
with open('models/SMOTED_RF_FIN_SBV_multi_2year_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# predict a single sample
def predict_a_sample_RF(x):
    y_pred = model.predict(x.reshape(1, -1))
    return y_pred
