import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dengue Dashboard", layout="wide")

st.title("Dengue Forecasting Dashboard")

# Upload dataset
file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Column selection
    column = st.selectbox("Select column to visualize", df.columns)

    # Plot
    st.subheader("Line Chart")
    fig, ax = plt.subplots()
    ax.plot(df[column])
    ax.set_title(column)
    st.pyplot(fig)

    # Basic stats
    st.subheader("Summary Statistics")
    st.write(df.describe())

else:
    st.info("Upload a dataset to start")
