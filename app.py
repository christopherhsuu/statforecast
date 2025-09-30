from shiny import App, render, ui
from prediction import load_data

batting = load_data()

app_ui = ui.page_fluid(
    ui.h2("StatForecast App"),
    ui.input_selectize(
        "player",
        "Choose a player:",
        choices=sorted(batting["Name"].unique())
    ),
    ui.output_table("player_stats")
)

def server(input, output, session):
    @output
    @render.table
    def player_stats():
        return batting[batting["Name"] == input.player()]

app = App(app_ui, server)
