# ==========================================
# Credit Card Fraud Detection
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("creditcard.csv")

print("="*50)
print("First Five Rows")
print("="*50)
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df["Class"].value_counts())

# ==========================================
# Data Visualization
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# Features and Target
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# Logistic Regression Model
# ==========================================

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# ==========================================
# Random Forest Model
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# ==========================================
# Evaluation Function
# ==========================================

def evaluate(model_name, y_true, y_pred):

    print("\n" + "="*50)
    print(model_name)
    print("="*50)

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(model_name + " Confusion Matrix")
    plt.show()

# ==========================================
# Evaluate Models
# ==========================================

evaluate(
    "Logistic Regression",
    y_test,
    lr_pred
)

evaluate(
    "Random Forest",
    y_test,
    rf_pred
)

# ==========================================
# Feature Importance
# ==========================================

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(8,6))
importance.head(10).plot(kind="bar")
plt.title("Top 10 Important Features")
plt.ylabel("Importance Score")
plt.show()

# ==========================================
# Predict New Transaction
# ==========================================

sample = X_test.iloc[[0]]

prediction = rf.predict(sample)

print("\nPrediction Result")

if prediction[0] == 1:
    print("⚠ Fraudulent Transaction")
else:
    print("✅ Genuine Transaction")