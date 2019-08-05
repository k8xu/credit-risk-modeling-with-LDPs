# Credit Risk Modeling with Low Default Portfolios
Financial credit risk assessment with low default portfolios (LDPs) is a difficult and open problem due to highly imbalanced datasets that contain significantly more good credit examples than bad ones. Current research uses machine learning methods, but an ideal classification model for credit scoring has yet to be discovered.

This project will examine state-of-the-art methods for approaching credit risk prediction, propose models that have demonstrated good performance in research, and apply the models to the German credit dataset. The code and writeup was initially done with Jupyter notebooks, so this repository will serve as a way to make my work more publicly available.


## State-of-the-Art Paper
A systematic search methodology was used to compile relevant papers on recent machine learning methods for imbalanced learning. The paper has been uploaded as a PDF file in this repository.


## Proposed Models
The models below have shown good performance in research and are evaluated on the German credit dataset.
1. Example-Dependent Cost-Sensitive Decision Trees
2. Cluster-Based Undersampling with Boosting
3. Extreme Gradient Boosting

The table below shows the author and type for each model.

| Method   | Author                                           | Notes                                                                   |
| -------- | ------------------------------------------------ | ----------------------------------------------------------------------- |
| ECSDT    | Bahnsen et al. (2015)                            | Bagging on decision trees                                               |
| CUSBoost | Rayhan et al. (2017)                             | Boosting with undersampling                                             |
| XGBoost  | Kuusik et al. (2018) and Mohammadi et al. (2019) | Boosting with Bayesian hyperparameter optimization but without sampling |


## Example-Dependent Cost-Sensitive Decision Trees (ECSDT)
ECSDT extends example-dependent cost-sensitive decision trees (CSDT) by creating random subsamples of the training set and training a CSDT on each one, and then the base classifiers are combined using different combination methods. Bahnsen et al. (2015) evaluated 12 different algorithms using four different random inducers and three different combinators. They determined using F1 score that random patches is the best random inducer and weighted voting is the best combination method.

[Bahnsen et al. (2015)](https://arxiv.org/pdf/1505.04637.pdf)

[GitHub](https://github.com/albahnsen/CostSensitiveClassification)

[Documentation](http://albahnsen.github.io/CostSensitiveClassification/)


## Cluster-Based Undersampling with Boosting (CUSBoost)
CUSBoost is a boosting algorithm that first clusters majority class instances into several groups using k-means, and then selects instances from each cluster using random undersampling. The model is similar to SMOTEBoost and RUSBoost, and it works best on highly clusterable data. Their modified sampling approach forms more representative samples to balance the dataset, and the number of clusters can be optimized using hyperparameter optimization. CUSBoost performed better on average using AUC compared to AdaBoost, RUSBoost, and SMOTEBoost on more imbalanced datasets.

[Rayhan et al. (2017)](https://arxiv.org/pdf/1712.04356.pdf)

[GitHub](https://github.com/farshidrayhanuiu/CUSBoost)

**Observations**

* The original algorithm fit for `pima.txt` from Rayhan et al. (2017) may be good for selecting majority class instances, but its overall performance needs improvement for imbalanced datasets. The range of scores for AUC is 0.60-0.70 and for MCC is 0.20-0.30.
* Main challenge: the confusion matrix shows that the model can classify the good credit class more correctly, but there are many bad credit risk examples that are misclassified as good credit than there are correctly classified bad credit examples.
* Changing the number of clusters and age categories, as well as feature selection, has shown slight improvement in the classification matrix. However, further parameter tuning and testing will need to be done for the model to be reliable and practical.

### Modified CUSBoost
The modified model adapts to `credit_data.csv` by extracting values from categorical features and adjusting to the CSV file format. A lesson learned is that metric functions must be computed within each split rather than on `y_test` and `y_pred` afterwards, or else only using values from last split instead of all the data. Similarly, `auc` is the AUC of the final split, so use `best_auc` which takes the highest AUC of all splits.

**Observations**

* SAMME achieves better AUC than SAMME.R in terms of average and best (+0.03).
* X Norm did slighly worse than original but not statistically significant (-0.01).
* Using fewer or more splits did not improve average or best AUC.
* Grouping Age by category slightly improved average AUC from original baseline.
* Scaling Credit amount by log also improved average AUC from original baseline.
* Using both Age_cat and log on Credit amount increased average AUC from 0.738 to 0.741.
* Changing the number of Clusters also slightly improved average AUC, though there is not a clear correlation between how many clusters and AUC.


## Extreme Gradient Boosting (XGBoost)
The proposed XGBoost model modifies the work of Mohammadi et al. (2019) by using Bayesian hyperparameter optimization but not ADASYN. Their experiments show that the original model performed better than Random Forest, Logistic Regression, and Support Vector Machine on TPR, TNR, FPR, FNR, and accuracy. Kuusik et al. (2018) found that resampling can result in many false positives, and they were able to achieve a 99.5% accuracy from XGBoost without additional sampling. The XGBoost documentation shows that the `scale_pos_weight` parameter also considers class imbalance, so data sampling is not used for this model.

[Kuusik et al. (2018)](https://www.researchgate.net/publication/325174759_Business_Credit_Scoring_of_Estonian_Organizations)

[Mohammadi et al. (2019)](https://www.researchgate.net/publication/331977568_Exploring_the_impact_of_foot-by-foot_track_geometry_on_the_occurrence_of_rail_defects)

[Documentation](https://xgboost.readthedocs.io/en/latest/)


## Notes
**Bias:** Difference between average prediction and correct output. Model with high bias pays little attention to training data and oversimplifies the model, resulting in high error on training and testing data.

**Variance:** Spread of predicted outputs for given data point. Model with high variance pays a lot of attention to training data and overfits the data, resulting in high error on testing data.

**Bias-variance tradeoff:** A model with few parameters may have high bias and low variance, while a model with many parameters may have high variance and low bias. An optimal model will find the balance between bias and variance and minimize the total error.

**How to install Python packages using `pip` in Jupyter notebook:** Add an exclamation mark (`!`) before the command. For example, to install `costcla` for ECSDT, type `!pip install costcla` and it should work.
