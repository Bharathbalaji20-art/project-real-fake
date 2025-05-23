import streamlit as st
import joblib
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
st.title(" Fake News Detector")


input_text = st.text_area("Enter a news article text below 👇", height=20)

if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        if prediction == "FAKE":
            st.error(" This news article is likely FAKE!")
        else:
            st.success(" This news article is likely REAL!")
