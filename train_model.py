import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset

# Load Dataset

df = pd.read_csv("dataset/drowsiness_dataset_new.csv")

print(df.columns)
print(df.head())

# Remove impossible head angles


df = df[
    (df["Pitch"].abs() <= 90) &
    (df["Roll"].abs() <= 90) &
    (df["Yaw"].abs() <= 90)
]

print("After Cleaning:", df.shape)


# Features & Labels


X = df[["EAR", "MAR", "Pitch", "Roll", "Yaw"]]
y = df["Label"]


# Encode Labels


encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Classes:", encoder.classes_)

# Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# Prediction


pred = model.predict(X_test)


# Evaluation


print("\nAccuracy :", accuracy_score(y_test, pred))

print("\nClassification Report\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, pred))


# Save Model

joblib.dump(model, "model/drowsiness_model.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("\nModel Saved Successfully!")