import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px


def main():
    st.title("Online Fraud Detection ")

    # Upload CSV file
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        st.write("File uploaded successfully.")

        # Read CSV file
        data = pd.read_csv(uploaded_file)

        # Sort data by name
        sorted_data = data.sort_values(by='nameOrig')

        # Display first few rows of the sorted dataset
        st.subheader("Sorted Dataset:")
        st.write(sorted_data.head())

        # Display options in the sidebar
        display_min_max_amount = st.sidebar.checkbox("Display Min and Max Amount")
        display_type_of_payment = st.sidebar.checkbox("Display Type of Payment")

        if display_min_max_amount:
            st.subheader("Minimum and Maximum Amount:")
            min_amount = sorted_data['amount'].min()
            max_amount = sorted_data['amount'].max()
            st.write(f"Minimum Amount: {min_amount}")
            st.write(f"Maximum Amount: {max_amount}")

        if display_type_of_payment:
            st.subheader("Type of Payment:")
            type_counts = sorted_data["type"].value_counts()
            st.write(type_counts)

        # Preprocessing steps
        data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
        data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

        # Display summary statistics
        st.subheader("Summary Statistics:")
        st.write(data.describe())

        # Display distribution of transaction type
        type_counts = data["type"].value_counts()
        transactions = type_counts.index
        quantity = type_counts.values

        fig = px.pie(values=quantity, names=transactions, hole=0.5, title="Distribution of Transaction Type")
        st.plotly_chart(fig)

        # Splitting the data
        x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
        y = np.array(data["isFraud"])
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

        # Training a machine learning model
        model = DecisionTreeClassifier()
        model.fit(xtrain, ytrain)
        accuracy = model.score(xtest, ytest)

        st.subheader("Model Accuracy:")
        st.write(f"Accuracy: {accuracy}")

        # Prediction
        sample_features = np.array([[4, 9000.60, 9000.60, 0.0]])  # Sample feature
        prediction = model.predict(sample_features)

        st.subheader("Prediction:")
        st.write(prediction)


if __name__ == "__main__":
    main()
