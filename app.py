from shiny import App, render, ui
from prediction import load_data, train_model, project_next_season

# Load data + train model once at startup
batting = load_data()
model = train_model(batting)

app_ui = ui.page_fluid(
    ui.h2("StatForecast App"),

    # Select a player
    ui.input_selectize(
        "player",
        "Choose a player:",
        choices=sorted(batting["Name"].unique())
    ),

    # Output: show projected stats
    ui.output_table("projection")
)

def server(input, output, session):

    @output
    @render.table
    def projection():
        # Run projection for the chosen player
        preds = project_next_season(input.player(), batting, model)

        if preds is None:
            return {"Error": ["No data found for this player"]}

        # Turn dictionary into a table-like object
        import pandas as pd
        df = pd.DataFrame(preds, index=[0])
        return df

app = App(app_ui, server)
