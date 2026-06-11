import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. වෙබ් ඇප් එකේ Title එක
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Species Predictor 🌸")
st.write("මලේ පෙති වල විස්තර ඇතුළත් කර මල් වර්ගය නිවැරදිව හඳුනාගන්න.")

# 2. Model එක load කරගැනීම
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 3. User ගෙන්Inputs ලබාගැනීම
st.subheader("Enter Flower Measurements:")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.3)

# 4. Predict Button එක සහ Output එක
if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f"The predicted species is: **{prediction[0].upper()}** 🎉")
