import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Adult Income Classification App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression","Decision Tree","KNN",
     "Naive Bayes","Random Forest","XGBoost"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    model = joblib.load(f"model/{model_choice}.pkl")
    scaler = joblib.load("model/scaler.pkl")
    
    X = scaler.transform(df)
    predictions = model.predict(X)
    
    st.subheader("Predictions")
    st.write(predictions)
    
    if "income" in df.columns:
        y_true = df["income"]
        st.subheader("Classification Report")
        st.text(classification_report(y_true, predictions))
        
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

