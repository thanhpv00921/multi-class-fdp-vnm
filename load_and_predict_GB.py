import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
import pickle

# load
with open('models/multiclass_FDP_VN_1year_GB.pkl', 'rb') as f:
    model = pickle.load(f)

# predict a single sample
def predict_a_sample_GB(x):
    y_pred = model.predict(x.reshape(1, -1))
    return y_pred
