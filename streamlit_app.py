import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prediction import project_next_season, load_data, train_model

st.title("StatForecast: Player Performance Predictor")
st.write("Upload player data and forecast expected performance based on historical metrics.")

# File upload
uploaded_file = st.file_uploader("Upload CSV data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # User controls
    selected_player = st.selectbox("Select a player", df["Name"].unique())

    # Generate prediction
    d = load_data()
    m = train_model(d)
    pred = project_next_season(df[df['Name'] == selected_player], d, m)
    st.metric(label="Predicted wOBA", value=round(pred, 3))

    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(df["xwOBA"], df["launch_angle"], alpha=0.6)
    ax.set_xlabel("xwOBA")
    ax.set_ylabel("Launch Angle (°)")
    st.pyplot(fig)
else:
    st.info("Please upload your dataset to start the analysis.")
