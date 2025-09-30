import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

pd.set_option('display.float_format', '{:.4f}'.format)

FEATURES = ['Age','PA','BB%','K%','HR','ISO','wOBA']
TARGETS  = ['AVG','SLG','OBP','OPS','wRC+','HR','ISO','wOBA']
REQUIRED = set(FEATURES + TARGETS + ['Name','Season'])

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    if 'Player' in df.columns and 'Name' not in df.columns:
        colmap['Player'] = 'Name'
    if 'Year' in df.columns and 'Season' not in df.columns:
        colmap['Year'] = 'Season'
    return df.rename(columns=colmap)

def load_data() -> pd.DataFrame:
    df = pd.read_csv("fangraphs_hitters_2010_2024.csv")
    df = _normalize_columns(df)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}\nHave: {df.columns.tolist()}")

    # basic filter
    df = df[df['PA'] >= 200].copy()

    # clean percent cols to fractions (0–1)
    for col in ['BB%', 'K%']:
        s = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors='coerce')
        df[col] = s / 100.0
    df = df.sort_values(['Name','Season']).copy()
    return df

def train_model(data: pd.DataFrame):
    # next-season targets via groupby shift
    work = data.copy()
    for col in TARGETS:
        work[f"Next_{col}"] = work.groupby('Name')[col].shift(-1)

    needed = FEATURES + [f"Next_{c}" for c in TARGETS]
    train = work.dropna(subset=needed)

    X = train[FEATURES]
    Y = train[[f"Next_{c}" for c in TARGETS]]

    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, Y)
    return model

def project_next_season(player_name: str, data: pd.DataFrame, model) -> dict | None:
    p = data[data['Name'] == player_name].sort_values('Season')
    if p.empty:
        return None
    latest = p.iloc[-1][FEATURES]
    features_df = pd.DataFrame([latest], columns=FEATURES)  # keep names
    preds = model.predict(features_df)[0]

    return {col: round(val, 3) for col, val in zip(TARGETS, preds)}

if __name__ == "__main__":
    # Optional quick local smoke test:
    d = load_data()
    m = train_model(d)
    print(project_next_season("Aaron Judge", d, m))
