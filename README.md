# Credit Risk Modeling with Low Default Portfolios
Financial credit risk assessment with low default portfolios (LDPs) is a difficult and open problem due to highly imbalanced datasets that contain significantly more good credit examples than bad ones. Current research uses machine learning methods, but an ideal classification model for credit scoring has yet to be discovered.

This project will examine state-of-the-art methods for approaching credit risk prediction, propose models that have demonstrated good performance in research, and apply the models to the German credit dataset. The code and writeup was initially done with Jupyter notebooks, so this repository will serve as a way to make my work more publicly available.


### State-of-the-Art Paper
A systematic search methodology was used to compile relevant papers on recent machine learning methods for imbalanced learning. The paper has been uploaded as a PDF file in this repository.


### Proposed Models
The models below have shown good performance in research and are evaluated on the German credit dataset.
1. Example-Dependent Cost-Sensitive Decision Trees
2. Cluster-Based Undersampling with Boosting
3. Extreme Gradient Boosting


### Notes
Bias: Difference between average prediction and correct output. Model with high bias pays little attention to training data and oversimplifies the model, resulting in high error on training and testing data.

Variance: Spread of predicted outputs for given data point. Model with high variance pays a lot of attention to training data and overfits the data, resulting in high error on testing data.

Bias-variance tradeoff: A model with few parameters may have high bias and low variance, while a model with many parameters may have high variance and low bias. An optimal model will find the balance between bias and variance and minimize the total error.
