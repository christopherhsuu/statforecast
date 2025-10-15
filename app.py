# app.py
from shiny import App, render, ui
import pandas as pd
import traceback

from prediction import load_data, train_model, project_next_season

INIT_ERROR = ""  # will show in UI if init fails

try:
    batting = load_data()
    model = train_model(batting)
    player_choices = sorted(pd.Series(batting["Name"]).dropna().unique().tolist())
except Exception:
    INIT_ERROR = traceback.format_exc()
    batting = pd.DataFrame()
    model = None
    player_choices = []

app_ui = ui.page_fluid(
    ui.h2("StatForecast"),
    ui.markdown("Select a player to see **projected next-season stats**."),

    # show init errors (missing columns, missing sklearn, etc.)
    ui.output_text_verbatim("init_error"),

    ui.input_selectize("player", "Choose a player:", choices=player_choices, multiple=False),
    ui.output_table("projection")
)

def server(input, output, session):

    @output
    @render.text
    def init_error():
        return INIT_ERROR  # empty string if ok

    @output
    @render.table
    def projection():
        if INIT_ERROR:
            return pd.DataFrame({"error": ["App failed during startup. See error above."]})
        if not input.player():
            return pd.DataFrame({"info": ["Pick a player"]})

        try:
            preds = project_next_season(input.player(), batting, model)
            if preds is None:
                return pd.DataFrame({"error": [f"No rows for '{input.player()}'"]})
            return pd.DataFrame(preds, index=[0])
        except Exception:
            tb = traceback.format_exc()
            return pd.DataFrame({"exception": [tb]})

app = App(app_ui, server)
