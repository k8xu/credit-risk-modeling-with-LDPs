import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import interp
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

dataset = 'credit_data.csv' # Used modified version from Kaggle, original from UCI Machine Learning Repository
print("dataset : ", dataset)
df = pd.read_csv(dataset)

# Drop first column containing original row numbers
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()

print("Age: ", credit_data['Age'].unique())
print("Sex: ", credit_data['Sex'].unique())
print("Job: ", credit_data['Job'].unique())
print("Housing: ", credit_data['Housing'].unique())
print("Saving accounts: ", credit_data['Saving accounts'].unique())
print("Checking account: ", credit_data['Checking account'].unique())
# print("Credit amount: ", credit_data['Credit amount'].unique())
# print("Duration: ", credit_data['Duration'].unique())
print("Purpose: ", credit_data['Purpose'].unique())
print("Risk: ", credit_data['Risk'].unique())

# One hot encoding function
def one_hot(df, nan = False):
    original = list(df.columns)
    category = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = category, dummy_na = nan, drop_first = True)
    new_columns = [c for c in df.columns if c not in original]
    return df, new_columns

# Feature extraction
df = df.merge(pd.get_dummies(df['Sex'], drop_first=True, prefix='Sex'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Housing'], drop_first=True, prefix='Housing'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=False, prefix='Saving'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Checking account"], drop_first=False, prefix='Checking'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Purpose'], drop_first=True, prefix='Purpose'), left_index=True, right_index=True)

del df['Sex']
del df['Housing']
del df['Saving accounts']
del df['Checking account']
del df['Purpose']

# Map outputs to 0 (good) or 1 (bad)
df = df.merge(pd.get_dummies(df.Risk, prefix='Risk'), left_index=True, right_index=True)
del df['Risk']
del df['Risk_good']

# Separate X and y of dataset
X = np.array(df.drop(['Risk_bad'], axis=1))
y = np.array(df['Risk_bad'])
print("X:", X, '\n')
# print("y:", y, '\n')

# Rescale feature values to decimals between 0 and 1
normalization_object = Normalizer()
X = normalization_object.fit_transform(X)
# X = X_norm
# print("X_norm:", X_norm)