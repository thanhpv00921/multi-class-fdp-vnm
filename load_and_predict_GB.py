import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
import pickle

# load
with open('models/multiclass_FDP_VN_1year_GB.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler_params.pkl', 'rb') as f:
    scaler_params = pickle.load(f)
mean = scaler_params['mean']
std = scaler_params['std']

# predict a single sample
def predict_a_sample_GB(x):
    print(mean)
    print(std)
    print(x)
    x = (x - mean) / std
    print(x)
    y_pred = model.predict(x.reshape(1, -1))
    return y_pred
