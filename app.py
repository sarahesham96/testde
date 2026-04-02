import streamlit as st
import pickle

st.set_page_config(page_title="NLP Predictor", page_icon="📝")

@st.cache_resource
def load_models():
    clf = pickle.load(open("nlp_model.pkl", "rb"))
    cv = pickle.load(open("tranform.pkl", "rb"))
    return clf, cv

clf, cv = load_models()

st.title("NLP Model Prediction")
st.write("Enter a message and get the model prediction.")

message = st.text_input("Message")

if st.button("Predict"):
    if message.strip():
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        st.success(f"Prediction: {my_prediction[0]}")
    else:
        st.warning("Please enter a message.")
