import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Load dataset
df = pd.read_csv("dataset/cleaned_complaints.csv")

X = df["Reports"]
y = df["Genre"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)


# Train model
model = LinearSVC()
model.fit(X_train, y_train)


# Get new complaint
title = input("Enter complaint title: ")
subject = input("Enter subject: ")
text = input("Enter complaint: ")


# Combine complaint
new_complaint = title + " " + subject + " " + text


# Convert new complaint to TF-IDF
new_data = vectorizer.transform([new_complaint])


# Predict category
prediction = model.predict(new_data)


print("\n-----------------------------")
print("Predicted Complaint Category:")
print(prediction[0])
print("-----------------------------")
