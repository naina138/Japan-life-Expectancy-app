import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load full clustered dataset
df = pd.read_csv("clustered_data.csv")

# Drop non-numeric columns if present (like 'Prefecture')
non_numeric_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_cols)

# Split into input and output
X = df.drop(columns=['Cluster'])
y = df['Cluster']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save train and test as CSV
train_df = X_train.copy()
train_df['Cluster'] = y_train
train_df.to_csv("train.csv", index=False)

test_df = X_test.copy()
test_df['Cluster'] = y_test
test_df.to_csv("test.csv", index=False)

print("✅ Cleaned and split saved: 'train.csv' and 'test.csv'")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved")
