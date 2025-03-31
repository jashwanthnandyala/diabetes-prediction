import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
diabetes_dataset = pd.read_csv("")

# Split data into features (X) and target (Y)
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Evaluate model performance
train_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
test_accuracy = accuracy_score(classifier.predict(X_test), Y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model and scaler
joblib.dump(classifier, "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")


