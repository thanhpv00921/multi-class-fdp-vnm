import numpy as np
from train_CSBE import CSBE

import pickle

# load
with open('models/CSBE_FIN_SBV_multi_2year_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# predict a single sample
def predict_a_sample_CSBE(x):
    y_pred = model.predict(x.reshape(1, -1))
    return y_pred
