import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 6: Load test data (20%)
test_df = pd.read_csv("test.csv")

# STEP 7: Prepare features and target
X_test = test_df.drop(columns=['Cluster'])
y_test = test_df['Cluster']

# Load the saved scaler and SVM model
scaler = joblib.load("scaler.pkl")
svm_model = joblib.load("svm_model.pkl")

# Scale the test features
X_test_scaled = scaler.transform(X_test)

# STEP 8: Predict on test data
y_pred = svm_model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - SVM Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
print("📋 Classification Report:\n")
print(classification_report(y_test, y_pred))
