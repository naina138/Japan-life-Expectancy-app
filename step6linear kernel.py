import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
from joblib import load
svm_model = load("svm_model.pkl")

# Load original train data to get feature names
train_df = pd.read_csv("train.csv")
feature_names = train_df.drop(columns=['Cluster']).columns

# Get feature coefficients for each class
coefs = svm_model.coef_

# Visualize top features per class
for i, class_label in enumerate(svm_model.classes_):
    plt.figure(figsize=(10, 4))
    coef_series = pd.Series(coefs[i], index=feature_names)
    coef_series.sort_values().plot(kind='barh', title=f"Feature Importance for Class {class_label}")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()
