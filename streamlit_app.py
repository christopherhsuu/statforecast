import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from prediction import load_data, train_model, load_model, project_next_season, build_features

MODEL_PATH = Path("models/rf_multioutput.joblib")
# Preferred: read MODEL_URL from environment so local runs (export MODEL_URL=...) work.
import os
MODEL_URL = os.environ.get('MODEL_URL')

# If present, prefer Streamlit secrets (used on Streamlit Cloud). Wrap in try/except
# because accessing st.secrets raises StreamlitSecretNotFoundError when no secrets
# file exists locally.
try:
    secret_val = None
    try:
        secret_val = st.secrets.get('MODEL_URL')
    except Exception:
        secret_val = None
    if secret_val:
        MODEL_URL = secret_val
except Exception:
    # be defensive: ignore any error coming from secrets parsing
    pass

def _download_file(url: str, dst: Path) -> None:
    """Download a file to dst (streamed)."""
    import requests
    dst.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dst, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


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

st.write("### Data preview (one row per player: career summary)")
# Show a per-player career summary instead of repeating seasons per player
if 'Name' in df.columns:
    player_summary = (
        df.groupby('Name', dropna=True)
        .agg(
            First_Season=('Season', 'min'),
            Last_Season=('Season', 'max'),
            Seasons_Played=('Season', lambda x: x.nunique()),
            Total_PA=('PA', 'sum'),
            Mean_wOBA=('wOBA', 'mean')
        )
        .reset_index()
        .sort_values(['Name'])
    )
    # show a limited preview but allow expanding
    st.dataframe(player_summary.head(200))
    if st.checkbox('Show full player list'):
        st.dataframe(player_summary)
else:
    st.dataframe(df.head())

# Model load or train
model = None
metadata = None
# Try to ensure model is present: if it's missing but MODEL_URL is provided, download it
if not MODEL_PATH.exists() and MODEL_URL and not retrain:
    try:
        with st.spinner('Downloading model...'):
            _download_file(MODEL_URL, MODEL_PATH)
        st.sidebar.info('Model downloaded')
    except Exception as e:
        st.sidebar.warning(f'Failed to download model: {e}')

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

# Player lookup: searchable dropdown
player_choices = sorted(df['Name'].dropna().unique().tolist())
selected_player = st.selectbox("Choose a player (type to search)", player_choices)

if selected_player:
    try:
        preds = project_next_season(selected_player, df, model)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        preds = None

    if preds is None:
        st.error("Player not found in dataset or prediction failed.")
    else:
        st.success(f"Predicted next-season stats for {selected_player}")
        cols = st.columns(4)
        i = 0
        for stat, val in preds.items():
            cols[i % 4].metric(label=stat, value=val)
            i += 1

        # show player's history plot for a couple of stats
        player_hist = df[df['Name'] == selected_player].sort_values('Season')
        if not player_hist.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(player_hist['Season'], player_hist['wOBA'], marker='o', label='wOBA')
            if 'ISO' in player_hist.columns:
                ax.plot(player_hist['Season'], player_hist['ISO'], marker='x', label='ISO')
            ax.set_xlabel('Season')
            ax.set_title(f'Historical wOBA / ISO for {selected_player}')
            ax.legend()
            st.pyplot(fig)
        else:
            st.info('No historical rows to plot for this player.')

        st.markdown('---')
        st.write('Model metadata')
        st.json(metadata)
