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