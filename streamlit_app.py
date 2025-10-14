import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from prediction import load_data, train_model, load_model, project_next_season, build_features

MODEL_PATH = Path("models/rf_multioutput.joblib")


st.set_page_config(page_title="StatForecast", layout="wide")
st.title("StatForecast: Next-Season Player Predictor")

with st.sidebar:
    st.header("Data & Model")
    upload = st.file_uploader("Upload CSV (optional)", type="csv")
    retrain = st.button("Retrain model")
    st.markdown("---")
    st.write("Model file: ")
    st.write(str(MODEL_PATH))

# Load data (uploaded takes precedence)
if upload:
    df = pd.read_csv(upload)
else:
    df = load_data()

st.write("### Data preview")
st.dataframe(df.head())

# Model load or train
model = None
metadata = None
if MODEL_PATH.exists() and not retrain:
    try:
        model, metadata = load_model(str(MODEL_PATH))
        st.sidebar.success(f"Loaded model (rmse={metadata.get('rmse'):.3f})")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

if retrain or model is None:
    with st.spinner('Training model... this may take a minute'):
        model, metadata = train_model(df, save_path=str(MODEL_PATH))
    st.sidebar.success(f"Model trained (rmse={metadata.get('rmse'):.3f})")

st.sidebar.markdown("---")
st.sidebar.write(f"Rows used for training: {metadata.get('train_rows', '?')}")

# Player lookup
player_input = st.text_input("Player name (exact match)")
search_btn = st.button("Predict")

if search_btn and player_input:
    preds = project_next_season(player_input, df, model)
    if preds is None:
        st.error("Player not found in dataset (check exact name).")
    else:
        st.success(f"Predicted next-season stats for {player_input}")
        cols = st.columns(4)
        i = 0
        for stat, val in preds.items():
            cols[i % 4].metric(label=stat, value=val)
            i += 1

        # show player's history plot for a couple of stats
        fe = build_features(df)
        player_hist = df[df['Name'] == player_input].sort_values('Season')
        if not player_hist.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(player_hist['Season'], player_hist['wOBA'], marker='o', label='wOBA')
            if 'ISO' in player_hist.columns:
                ax.plot(player_hist['Season'], player_hist['ISO'], marker='x', label='ISO')
            ax.set_xlabel('Season')
            ax.set_title(f'Historical wOBA / ISO for {player_input}')
            ax.legend()
            st.pyplot(fig)
        else:
            st.info('No historical rows to plot for this player.')

        st.markdown('---')
        st.write('Model metadata')
        st.json(metadata)

else:
    st.info('Enter a player name (exact match) and press Predict.')
