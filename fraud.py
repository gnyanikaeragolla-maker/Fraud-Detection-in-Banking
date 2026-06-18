# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("bank_fraud_data.csv")

# Convert text columns into numbers
encoder = LabelEncoder()
data["Location"] = encoder.fit_transform(data["Location"])
data["TransactionType"] = encoder.fit_transform(data["TransactionType"])

# Input and Output
X = data.drop(["TransactionID", "Fraud"], axis=1)
y = data["Fraud"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Fraud Detection Accuracy:", accuracy)