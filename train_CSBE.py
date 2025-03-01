import numpy as np
import pandas as pd
from scipy import stats
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from scipy.stats import mode

# Define the CSBE class
class CSBE:
    def __init__(self, base_estimator=None, n_estimators=10, sampling_mode=1):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=5, max_features=1)
        self.n_estimators = n_estimators
        self.base_learners = []
        self.is_fitted = False

    def fit(self, X, y):
        random.seed(42)
        
        n_samples = X.shape[0]
        X_0 = []
        X_1 = []
        X_2 = []
        y_0 = []
        y_1 = []
        y_2 = []
        
        # split training set by labels to have a set of distress and non-distress samples.
        for i in range(n_samples):
            if y[i] == 0:
                X_0.append(X[i])
                y_0.append(y[i])
            elif y[i] == 1:
                X_1.append(X[i])
                y_1.append(y[i])
            elif y[i] == 2:
                X_2.append(X[i])
                y_2.append(y[i])
        
        X_0 = np.asarray(X_0)
        X_1 = np.asarray(X_1)
        X_2 = np.asarray(X_2)
        y_0 = np.asarray(y_0)
        y_1 = np.asarray(y_1)
        y_2 = np.asarray(y_2)
        
        # calculate no of non-distress samples.
        non_distress_no = X_0.shape[0]
        #print(non_distress_no)
        
        self.base_learners = []

        for _ in range(self.n_estimators):
            # take a subset of non-distress samples with the same size as distressed samples.
            indices = np.random.choice(range(X_0.shape[0]), size=non_distress_no, replace=True)
            X_non_distress, y_non_distress = X_0[indices], y_0[indices]

            indices = np.random.choice(range(X_1.shape[0]), size=non_distress_no, replace=True)
            X_distress, y_distress = X_1[indices], y_1[indices]
            X_sampled = np.concatenate((X_non_distress, X_distress), axis=0)
            y_sampled = np.concatenate((y_non_distress, y_distress), axis=0)
            
            indices = np.random.choice(range(X_2.shape[0]), size=non_distress_no, replace=True)
            X_distress, y_distress = X_2[indices], y_2[indices]
            X_sampled = np.concatenate((X_sampled, X_distress), axis=0)
            y_sampled = np.concatenate((y_sampled, y_distress), axis=0)
            
            #print('X_sampled.shape: ', X_sampled.shape)
            #print('y_sampled.shape: ', y_sampled.shape)

            cloned_estimator = clone(self.base_estimator)
            cloned_estimator.fit(X_sampled, y_sampled)
            self.base_learners.append(cloned_estimator)
        
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise Exception("This CSBE instance is not fitted yet.")
        
        predictions = np.array([learner.predict(X) for learner in self.base_learners]).T
        final_predictions, _ = mode(predictions, axis=1)
        return final_predictions.ravel()
