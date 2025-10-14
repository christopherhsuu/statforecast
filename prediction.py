import os
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

pd.set_option('display.float_format', '{:.4f}'.format)

# Core configuration
BASE_CSV = "fangraphs_hitters_2010_2024.csv"
FEATURES = ['Age', 'PA', 'BB%', 'K%', 'HR', 'ISO', 'wOBA']
TARGETS = ['AVG', 'SLG', 'OBP', 'OPS', 'wRC+', 'HR', 'ISO', 'wOBA']


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    if 'Player' in df.columns and 'Name' not in df.columns:
        colmap['Player'] = 'Name'
    if 'Year' in df.columns and 'Season' not in df.columns:
        colmap['Year'] = 'Season'
    return df.rename(columns=colmap)


def load_data(path: str = BASE_CSV, min_pa: int = 200) -> pd.DataFrame:
    """Load the Fangraphs CSV and perform light cleaning.

    Returns a DataFrame sorted by Name, Season.
    """
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # ensure required columns exist (be permissive)
    required = set(FEATURES + TARGETS + ['Name', 'Season'])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}\nHave: {df.columns.tolist()}")

    # convert percent strings like '12.3%' to float 0.123
    for col in ['BB%', 'K%']:
        if df[col].dtype == object:
            s = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors='coerce')
            df[col] = s / 100.0

    # coarse filter
    df = df[df['PA'] >= min_pa].copy()
    df = df.sort_values(['Name', 'Season']).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Create lag features and rolling means for each player.

    Produces columns: Age, PA, BB%, K%, HR, ISO, wOBA plus
    Last1_{stat}, Last2_{stat}, ... and RollMean3_{stat} for TARGETS.
    """
    work = df.copy()
    for lag in range(1, lags + 1):
        for stat in TARGETS + ['PA', 'Age']:
            work[f'lag{lag}_{stat}'] = work.groupby('Name')[stat].shift(lag)

    # rolling mean of last `lags` seasons for targets
    for stat in TARGETS:
        work[f'roll{lags}_{stat}'] = work.groupby('Name')[stat].shift(1).rolling(window=lags, min_periods=1).mean().reset_index(level=0, drop=True)

    # drop rows without next-season target
    for t in TARGETS:
        work[f'Next_{t}'] = work.groupby('Name')[t].shift(-1)

    return work


def _assemble_matrix(fe_df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if feature_columns is None:
        # select base features + last-season stats + rolling means that exist
        candidate = []
        for f in FEATURES:
            if f in fe_df.columns:
                candidate.append(f)
        # include last season's target stats if present
        for stat in TARGETS:
            col = f'lag1_{stat}'
            if col in fe_df.columns:
                candidate.append(col)
            rm = [c for c in fe_df.columns if c.startswith('roll') and c.endswith(f'_{stat}')]
            candidate += rm

        feature_columns = candidate

    needed_targets = [f'Next_{t}' for t in TARGETS]
    matrix = fe_df.dropna(subset=feature_columns + needed_targets).copy()
    X = matrix[feature_columns]
    Y = matrix[needed_targets].rename(columns={f'Next_{t}': t for t in TARGETS})
    return X, Y


def train_model(df: pd.DataFrame, save_path: Optional[str] = None, random_state: int = 42) -> Tuple[MultiOutputRegressor, Dict]:
    """Train a RandomForest multi-output model and optionally save it.

    Returns (model, metadata) where metadata contains feature columns and metrics.
    """
    fe = build_features(df, lags=3)
    X, Y = _assemble_matrix(fe)

    # keep feature list for later
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=random_state)

    base = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    # older scikit-learn versions may not support `squared=False` keyword; compute RMSE
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metadata = {
        'feature_columns': feature_columns,
        'rmse': float(rmse),
        'train_rows': X_train.shape[0],
        'test_rows': X_test.shape[0],
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({'model': model, 'metadata': metadata}, save_path)

    return model, metadata


def load_model(path: str):
    payload = joblib.load(path)
    return payload['model'], payload.get('metadata', {})


def _predict_with_uncertainty(model: MultiOutputRegressor, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return predictions and a simple uncertainty estimate (std across estimators per output).

    For MultiOutputRegressor wrapping RandomForestRegressor, we compute per-output std
    across the underlying estimators for that output.
    """
    preds = model.predict(X)

    # Attempt to compute per-output std using sub-estimators if available
    try:
        # model.estimators_ is a list of estimators (one per output) when using MultiOutputRegressor
        per_output_stds = []
        for out_idx, est in enumerate(model.estimators_):
            # est is a RandomForestRegressor; use its estimators_ (trees)
            trees = getattr(est, 'estimators_', None)
            if not trees:
                per_output_stds.append(np.zeros(preds.shape[1]))
                continue
            tree_preds = np.vstack([t.predict(X) for t in trees])  # (n_trees, n_samples)
            std_per_sample = tree_preds.std(axis=0)
            per_output_stds.append(std_per_sample)

        # per_output_stds shape: (n_outputs, n_samples)
        per_output_stds = np.vstack(per_output_stds).T  # (n_samples, n_outputs)
    except Exception:
        per_output_stds = None

    return preds, per_output_stds


def predict_player(player_name: str, df: pd.DataFrame, model: MultiOutputRegressor) -> Optional[Dict[str, float]]:
    """Predict next-season TARGETS for the given player using the trained model.

    Returns a dict of {stat: predicted_value} or None if player not found.
    """
    fe = build_features(df, lags=3)
    # find latest row for player
    p = fe[fe['Name'] == player_name].sort_values('Season')
    if p.empty:
        return None
    latest = p.iloc[-1]

    # build X using model's expected features if possible
    # try to infer feature columns from model if it was trained with metadata
    feature_columns = None
    if hasattr(model, 'estimators_'):
        # can't reliably extract feature names from sklearn pipelines here; fallback to FEATURES + lag1_*
        feature_columns = [c for c in fe.columns if (c in FEATURES) or c.startswith('lag1_') or c.startswith('roll')]

    X = pd.DataFrame([latest[feature_columns]]) if feature_columns else pd.DataFrame([latest[FEATURES]])
    preds, uncert = _predict_with_uncertainty(model, X)
    preds = preds[0]

    result = {stat: float(round(val, 3)) for stat, val in zip(TARGETS, preds)}
    return result


def project_next_season(player_name: str, data: pd.DataFrame, model) -> dict | None:
    # Backwards-compatible alias
    return predict_player(player_name, data, model)


if __name__ == '__main__':
    # quick smoke test
    d = load_data()
    m, meta = train_model(d)
    print('Trained; RMSE:', meta['rmse'])
    sample = d['Name'].dropna().unique()[0]
    print('Sample prediction for', sample, project_next_season(sample, d, m))
