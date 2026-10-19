import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="College Complaint Classifier",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎓 College Complaint Management System")

st.write(
    "Enter a student complaint and the Machine Learning model "
    "will predict its category."
)


# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv("dataset/cleaned_complaints.csv")

    # Input and target
    X = df["Reports"].fillna("")
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
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(X_train)

    # Train model
    model = LinearSVC()
    model.fit(X_train, y_train)

    return model, vectorizer


# Train model
model, vectorizer = train_model()


# -----------------------------
# Complaint Input
# -----------------------------

st.subheader("📝 Enter Complaint")

title = st.text_input(
    "Complaint Title",
    placeholder="Example: WiFi Problem"
)

subject = st.text_input(
    "Subject",
    placeholder="Example: Internet Connection"
)

text = st.text_area(
    "Complaint Description",
    placeholder="Describe your complaint here...",
    height=150
)


# -----------------------------
# Prediction Button
# -----------------------------

if st.button("🔍 Predict Complaint Category"):

    if text.strip() == "":
        st.warning("Please enter a complaint description.")

    else:

        # Combine input
        new_complaint = (
            title + " " +
            subject + " " +
            text
        )

        # Convert to TF-IDF
        new_data = vectorizer.transform(
            [new_complaint]
        )

        # Predict
        prediction = model.predict(new_data)

        # Display result
        st.success("Complaint classified successfully!")

        st.subheader("🎯 Predicted Category")

        st.info(prediction[0])