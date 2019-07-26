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
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns


dataset = 'credit_data.csv'
print("dataset : ", dataset)
df = pd.read_csv(dataset)

# Drop first column containing original row numbers
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()

print("Age: ", df['Age'].unique())
print("Sex: ", df['Sex'].unique())
print("Job: ", df['Job'].unique())
print("Housing: ", df['Housing'].unique())
print("Saving accounts: ", df['Saving accounts'].unique())
print("Checking account: ", df['Checking account'].unique())
# print("Credit amount: ", df['Credit amount'].unique())
# print("Duration: ", df['Duration'].unique())
print("Purpose: ", df['Purpose'].unique())
print("Risk: ", df['Risk'].unique())

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

# K-fold cross validation that splits data into train and test set
skf = StratifiedKFold(n_splits=10, shuffle=True) # default n_splits is 5

top_auc = 0
mean_fpr = np.linspace(0, 1, 100) # np.linspace returns 100 evenly spaced numbers over interval [0,1]
number_of_clusters = 5 # Why did they choose 23?
percentage_to_choose_from_each_cluster = 0.5 # Undersampling ratio of 0.5 for each majority class cluster

for depth in range(2, 20, 10): # What is depth and estimators?
    for estimators in range(20, 50, 10):

        current_param_auc = []
        current_param_aupr = []
        tprs = []

        for train_index, test_index in skf.split(X, y):
            # print('train_index:', train_index)
            # print('test_index:', test_index)

            X_train = X[train_index]
            X_test = X[test_index]
            # print('X_train:', X_train)
            # print('X_test:', X_test)
            
            y_train = y[train_index]
            y_test = y[test_index]
            # print('y_train:', y_train)
            # print('y_test:', y_test)
            
            value, counts = np.unique(y_train, return_counts=True)
            minority_class = value[np.argmin(counts)]
            majority_class = value[np.argmax(counts)]

            idx_min = np.where(y_train == minority_class)[0]
            idx_maj = np.where(y_train == majority_class)[0]

            majority_class_instances = X_train[idx_maj]
            majority_class_labels = y_train[idx_maj]

            kmeans = KMeans(n_clusters=number_of_clusters)
            kmeans.fit(majority_class_instances)

            X_maj = []
            y_maj = []

            points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

            for key in points_under_each_cluster.keys():
                points_under_this_cluster = np.array(points_under_each_cluster[key])
                number_of_points_to_choose_from_this_cluster = math.ceil(
                    len(points_under_this_cluster) * percentage_to_choose_from_each_cluster)
                selected_points = np.random.choice(points_under_this_cluster,
                                                   size=number_of_points_to_choose_from_this_cluster)
                X_maj.extend(majority_class_instances[selected_points])
                y_maj.extend(majority_class_labels[selected_points])

            X_sampled = np.concatenate((X_train[idx_min], np.array(X_maj)))
            y_sampled = np.concatenate((y_train[idx_min], np.array(y_maj)))

            # Use AdaBoost as ensemble classifier of Decision Trees
            classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=depth),
                n_estimators=estimators,
                learning_rate=1, algorithm='SAMME.R') # SAMME is a discrete boosting algorithm
            # classifier = AdaBoostClassifier(
            #     DecisionTreeClassifier(max_depth=depth),
            #     n_estimators=estimators,
            #     learning_rate=1, algorithm='SAMME.R') # Achieved better AUC (+0.15) and MCC (+0.30)!

            classifier.fit(X_sampled, y_sampled)

            predictions = classifier.predict_proba(X_test)
            y_pred = classifier.predict(X_test)

            auc = roc_auc_score(y_test, predictions[:, 1])

            aupr = average_precision_score(y_test, predictions[:, 1])

            current_param_auc.append(auc)

            current_param_aupr.append(aupr)

            fpr, tpr, thresholds = roc_curve(y_test, predictions[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

        current_mean_auc = np.mean(np.array(current_param_auc))
        current_mean_aupr = np.mean(np.array(current_param_aupr))

        if top_auc < current_mean_auc:
            top_auc = current_mean_auc

            best_depth = depth
            best_estimators = estimators

            best_auc = top_auc
            best_aupr = current_mean_aupr

            best_tpr = np.mean(tprs, axis=0)
            best_fpr = mean_fpr

            best_precision, best_recall, _ = precision_recall_curve(y_test, predictions[:, 1])
            best_fpr, best_tpr, thresholds = roc_curve(y_test, predictions[:, 1])

print('plotting', dataset)
# plt.clf()
plt.plot(best_recall, best_precision, lw=2, color='Blue',
         label='Precision-Recall Curve')
plt.plot(best_fpr, best_tpr, lw=2, color='red',
         label='ROC curve')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="upper right")
plt.show()

# Evaluate model with metrics
print("AUC:", auc) # Higher is better
print("MCC:", matthews_corrcoef(y_test, y_pred)) # Closer to 1 is better
print("F1 Score:", f1_score(y_test, y_pred)) # Closer to 1 is better
print("Accuracy:", accuracy_score(y_test, y_pred)) # Closer to 1 is better

# plt.plot(fpr_c[1], tpr_c[1], lw=2, color='red',label='Roc curve: Clustered sampling') # Error: says fpr_c doesn't exist