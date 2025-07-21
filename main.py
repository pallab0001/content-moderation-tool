# content_moderation_tool.py

import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

# Sample dataset (can be replaced with a real one)
data = {
    'text': [
        "I hate this product",
        "You are so stupid",
        "Have a nice day",
        "Kill them all",
        "Thank you for your support",
        "I love this community"
    ],
    'label': ["offensive", "offensive", "clean", "offensive", "clean", "clean"]
}
df = pd.DataFrame(data)

# Split data
X = df['text']
y = df['label']

# Train a simple text classification model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Streamlit App
st.title("Automated Content Moderation Tool")
st.write("Enter a comment or post content to check if it's appropriate.")

user_input = st.text_area("Content to check:")

if user_input:
    prediction = model.predict([user_input])[0]
    st.write(f"\n### Result: **{prediction.upper()}**")

    if prediction == "offensive":
        st.error("This content may violate community guidelines.")
    else:
        st.success("This content appears to be clean.")
