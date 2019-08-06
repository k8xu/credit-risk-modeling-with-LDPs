"""
Make sure you have 'xgboost' and 'bayesian-optimization' installed using 'pip' before running the following code
"""
# Import packages, read credit_data.csv, drop column of row numbers
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from xgboost import XGBClassifier
from scipy import interp
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

dataset = 'credit_data.csv'
print("dataset : ", dataset)
df = pd.read_csv(dataset)

df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())


# One hot encoding function
def one_hot(df, nan = False):
    original = list(df.columns)
    category = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = category, dummy_na = nan, drop_first = True)
    new_columns = [c for c in df.columns if c not in original]
    return df, new_columns

# Feature extraction
df = df.merge(pd.get_dummies(df['Sex'], drop_first=True, prefix='Sex'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Housing'], drop_first=False, prefix='Housing'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=False, prefix='Saving'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Checking account"], drop_first=False, prefix='Checking'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Purpose'], drop_first=False, prefix='Purpose'), left_index=True, right_index=True)

# Group age into categories
interval = (18, 25, 40, 65, 100)
categories = ['Student', 'Younger', 'Older', 'Senior']
df["Age_cat"] = pd.cut(df.Age, interval, labels=categories)
df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=False, prefix='Age_cat'), left_index=True, right_index=True)

del df['Sex']
del df['Housing']
del df['Saving accounts']
del df['Checking account']
del df['Purpose']
del df['Age']
del df['Age_cat']

# Scale credit amount by natural log function
df['Credit amount'] = np.log(df['Credit amount'])

# Map outputs to 0 (good) or 1 (bad)
df = df.merge(pd.get_dummies(df.Risk, prefix='Risk'), left_index=True, right_index=True)
del df['Risk']
del df['Risk_good']
# print(df.head())


from bayes_opt import BayesianOptimization

# Separate X and y of dataset
X = np.array(df.drop(['Risk_bad'], axis=1))
y = np.array(df['Risk_bad'])

# Split train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Test on unseen data
xgboosted = XGBClassifier(learning_rate=0.2, 
                          gamma=5, 
                          n_estimators=1000, # int
                          max_depth=3, # int
                          min_child_weight=20, # int
                          subsample=0.94, 
                          colsample_bytree=0.95, 
                          scale_pos_weight=0.94)
print("XGBoosted :", xgboosted)

xgboosted.fit(X_train, y_train)
predictions = xgboosted.predict_proba(X_test)
y_pred = xgboosted.predict(X_test)

# Evaluate with AUC and MCC
auc = roc_auc_score(y_test, predictions[:, 1])
mcc = matthews_corrcoef(y_test, y_pred)
print("AUC :", auc)
print("MCC :", mcc)

# Other metrics
acc = accuracy_score(y_test, y_pred)
print("Accuracy :", acc)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print("Balanced accuracy :", bal_acc)


# Create confusion matrix
matrix = confusion_matrix(y_test, y_pred)
values = np.unique(y_pred)
sns.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False, xticklabels=values, yticklabels=values)
plt.xlabel('Truth')
plt.ylabel('Predicted')


# Create XGBoost model with Bayesian optimization
skf = StratifiedKFold(n_splits=5, shuffle=True)

all_auc = []
all_mcc = []
all_acc = []
all_bal_acc = []

def xgb_function(learning_rate, gamma, min_child_weight, subsample, colsample_bytree, scale_pos_weight):
    """
    Function with XGBoost parameters that returns AUC on train and test set
    """
    xgbclf = XGBClassifier(learning_rate=learning_rate, 
                           gamma=gamma, 
                           n_estimators=1000, 
                           max_depth=3, 
                           min_child_weight=min_child_weight, 
                           subsample=subsample, 
                           colsample_bytree=colsample_bytree, 
                           scale_pos_weight=scale_pos_weight)
    
    
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fun = X_train[train_index]
        y_train_fun = y_train[train_index]
        X_val = X_train[val_index]
        y_val = y_train[val_index]

        xgbclf.fit(X_train_fun, y_train_fun)
        predictions = xgbclf.predict_proba(X_val)
        y_pred = xgbclf.predict(X_val)
    
        auc = roc_auc_score(y_val, predictions[:, 1])
        mcc = matthews_corrcoef(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        
        all_auc.append(auc)
        all_mcc.append(mcc)
        all_acc.append(acc)
        all_bal_acc.append(bal_acc)
    
    
    mean_auc = np.mean(np.array(all_auc))
    mean_mcc = np.mean(np.array(all_mcc))
    mean_acc = np.mean(np.array(all_acc))
    mean_bal_acc = np.mean(np.array(all_bal_acc))

    return mean_auc
    
# Parameter bounds
pbounds = {'learning_rate': (0.01, 0.2), 
           'gamma': (1.0, 5.0), 
           'min_child_weight': (0, 20), 
           'subsample': (0.8, 1.0), 
           'colsample_bytree': (0.7, 1.0), 
           'scale_pos_weight': (0.5, 1.0)}
optimizer = BayesianOptimization(f=xgb_function, pbounds=pbounds, verbose=2)
optimizer.maximize(init_points=2, n_iter=3)
print("Optimizer :", optimizer.max)
