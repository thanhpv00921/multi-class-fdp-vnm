import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE
import pickle

data = pd.read_csv("dataset/vnm_enterprises-1year-multiclass-fdp.csv")
arr = data.to_numpy()

# Split predictors and true label.
colxx = 23
X = arr[:,2:colxx]
Y = arr[:,colxx:colxx+1]
X = X.astype(float)

# data normalization for predictors
X = stats.zscore(X, axis=0)

# split into train set and test set
rowxx = 11656 # train: (2010-2021) first 11656 rows; test: (2022) the remaining rows.
X_train = X[0:rowxx,:]
y_train = Y[0:rowxx,:]

X_test = X[rowxx:,:]
y_test = Y[rowxx:,:]

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# print numpy array shape to check the dimensions
print(X_train.shape) 
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape) 

# intialize a LogisticRegression model ---------------------------------------------------------------------------------
print('Multinomial Logistic Regression')
model = LogisticRegression(max_iter=1000, C=100, penalty='l2', solver='lbfgs')
model.fit(X_train, y_train)

# calculate accuracy_score, precision, recall, F1
y_pred = model.predict(X_test)
print(y_pred[:20])

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on Test:', accuracy)

precision = precision_score(y_test, y_pred, average='macro')
print('Precision on Test:', precision)

recall = recall_score(y_test, y_pred, average='macro')
print('Recall on Test:', recall)

f1 = f1_score(y_test, y_pred, average='macro')
print('F1 on Test:', f1)

# save
with open('models/multiclass_FDP_VN_1year_MLR.pkl','wb') as f:
    pickle.dump(model,f)
print('--------------------------------------')

# intialize a DecisionTreeClassifier model ---------------------------------------------------------------------------------
print('Decision Tree Classifier')
model = DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', criterion='gini', min_impurity_decrease=0.0)
# train the model
model.fit(X_train, y_train)

# calculate accuracy_score, precision, recall, F1
y_pred = model.predict(X_test)
print(y_pred[:20])

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on Test:', accuracy)

precision = precision_score(y_test, y_pred, average='macro')
print('Precision on Test:', precision)

recall = recall_score(y_test, y_pred, average='macro')
print('Recall on Test:', recall)

f1 = f1_score(y_test, y_pred, average='macro')
print('F1 on Test:', f1)

# save
with open('models/multiclass_FDP_VN_1year_DT.pkl','wb') as f:
    pickle.dump(model,f)
print('--------------------------------------')

# intialize a Support Vector Classifier model ---------------------------------------------------------------------------------
print('Support Vector Classifier')
model = SVC(kernel='rbf', C=100.0, gamma=0.1)
model.fit(X_train, y_train)

# calculate accuracy_score, precision, recall, F1
y_pred = model.predict(X_test)
print(y_pred[:20])

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on Test:', accuracy)

precision = precision_score(y_test, y_pred, average='macro')
print('Precision on Test:', precision)

recall = recall_score(y_test, y_pred, average='macro')
print('Recall on Test:', recall)

f1 = f1_score(y_test, y_pred, average='macro')
print('F1 on Test:', f1)

# save
with open('models/multiclass_FDP_VN_1year_SVM.pkl','wb') as f:
    pickle.dump(model,f)
print('--------------------------------------')

# intialize a RandomForestClassifier model ---------------------------------------------------------------------------------
print('Random Forest Classifier')
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=False)
# train the model
model.fit(X_train, y_train)

# calculate accuracy_score, precision, recall, F1
y_pred = model.predict(X_test)
print(y_pred[:20])

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on Test:', accuracy)

precision = precision_score(y_test, y_pred, average='macro')
print('Precision on Test:', precision)

recall = recall_score(y_test, y_pred, average='macro')
print('Recall on Test:', recall)

f1 = f1_score(y_test, y_pred, average='macro')
print('F1 on Test:', f1)

# save
with open('models/multiclass_FDP_VN_1year_RF.pkl','wb') as f:
    pickle.dump(model,f)
print('--------------------------------------')

# intialize a GradientBoostingClassifier model ---------------------------------------------------------------------------------
print('Gradient Boosting Classifier')
model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.01, max_depth=None, min_samples_split=2, min_samples_leaf=1, subsample=1.0, max_features='sqrt')
# train the model
model.fit(X_train, y_train)

# calculate accuracy_score, precision, recall, F1
y_pred = model.predict(X_test)
print(y_pred[:20])

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on Test:', accuracy)

precision = precision_score(y_test, y_pred, average='macro')
print('Precision on Test:', precision)

recall = recall_score(y_test, y_pred, average='macro')
print('Recall on Test:', recall)

f1 = f1_score(y_test, y_pred, average='macro')
print('F1 on Test:', f1)

# save
with open('models/multiclass_FDP_VN_1year_GB.pkl','wb') as f:
    pickle.dump(model,f)