import numpy as np
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
import pickle

# load trained model.
with open('models/multiclass_FDP_VN_1year_RF.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler_params.pkl', 'rb') as f:
    scaler_params = pickle.load(f)
mean = scaler_params['mean']
std = scaler_params['std']

# predict a single sample
def predict_a_sample_RF(x):
    x = (x - mean) / std
    y_pred = model.predict(x.reshape(1, -1))
    return y_pred
