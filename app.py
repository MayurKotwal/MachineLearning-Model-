import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Title and Description
st.title("Crime Prediction ML Models")
st.write("This app allows you to upload data and compare Logistic Regression, Random Forest, and Decision Tree models.")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())
    
    # Process data
    st.write("Processing the dataset...")
    data['Total_Losses'] = data.filter(like='Losses').sum(axis=1)
    data['Outcome'] = pd.cut(data['Total_Losses'], bins=[0, 5e7, 1e8, float('inf')], labels=['Low', 'Medium', 'High'])
    
    encoder = LabelEncoder()
    data['Country'] = encoder.fit_transform(data['Country'])
    
    X = data.drop(columns=['Outcome', 'Total_Losses'])
    y = data['Outcome']
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    st.write("Dataset processed successfully!")

    # Step 2: Model Selection
    model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Random Forest", "Decision Tree"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    else:
        model = DecisionTreeClassifier(random_state=42)

    # Train and Evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model: {model_choice}")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Visualize Predictions
    st.write("### Sample Predictions")
    results = pd.DataFrame({"Actual": y_test[:10].values, "Predicted": y_pred[:10]})
    st.write(results)

# Footer
st.write("Developed using Streamlit for machine learning deployment!")
