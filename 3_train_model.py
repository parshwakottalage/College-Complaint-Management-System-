import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_complaints.csv")

# Input
X = df["Reports"]

# Output/target
y = df["Genre"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


print("TF-IDF conversion completed!")


# Create ML model
model = LogisticRegression(max_iter=1000)


# Train model
model.fit(X_train, y_train)

print("Model training completed!")


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))