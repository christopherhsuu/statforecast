import pandas as pd
batting = pd.read_csv("fangraphs_hitters_2010_2024.csv")
batting = batting[batting['PA'] >= 200].copy()
for col in ['BB%', 'K%']:
    batting[col] = pd.to_numeric(batting[col].astype(str).str.rstrip('%'), errors='coerce') / 100
import pandas as pd
import numpy as np

pd.set_option('display.float_format', '{:.4f}'.format)

def get_career_stats(player_name, data):
    p = data[data['Name'] == player_name].copy()
    if p.empty:
        return None

    for col in ['BB%', 'K%']:
        s = pd.to_numeric(p[col].astype(str).str.rstrip('%'), errors='coerce')

        if s.quantile(0.98) > 1.5:
            s = s / 100.0
        if s.max() < 0.05:
            s = s * 100.0

        p[col] = s

    totals = p[['PA','HR','R','RBI','SB']].sum(numeric_only=True)
    rates  = p[['BB%','K%','ISO','wOBA']].mean(numeric_only=True)
    out = pd.concat([totals, rates])
    
    return out.round(4)

feature_cols = ['Age','PA','BB%','K%','HR','ISO','wOBA']
target_cols  = ['AVG','SLG','OBP','OPS','wRC+','HR','ISO','wOBA']
batting = batting.sort_values(['Name','Season']).copy()

for col in target_cols:
    batting[f"Next_{col}"] = batting.groupby('Name')[col].shift(-1)

train = batting.dropna(subset=feature_cols + [f"Next_{col}" for col in target_cols])

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

X = train[feature_cols]
Y = train[[f"Next_{col}" for col in target_cols]]

model = MultiOutputRegressor(LinearRegression())
model.fit(X, Y)

def project_next_season(player_name, data, model):
    p = data[data['Name'] == player_name].sort_values('Season')
    if p.empty:
        return None
    
    latest = p.iloc[-1][feature_cols]
    features_df = pd.DataFrame([latest], columns=feature_cols)
    
    preds = model.predict(features_df)[0]  # array of predictions
    
    # Map each prediction back to its stat
    return {col: round(val, 3) for col, val in zip(target_cols, preds)}

# Interactive prompt
player = "Aaron Judge"  # Example player name

# Get projection
projection = project_next_season(player, batting, model)

# Print results nicely
if projection is None:
    print(f"No data found for {player}")
else:
    print(f"\nProjection for {player} next season:")
    for stat, value in projection.items():
        print(f"{stat:5}: {value}")