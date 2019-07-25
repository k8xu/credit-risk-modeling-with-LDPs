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