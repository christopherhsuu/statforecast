from shiny import App, render, ui
from prediction import load_data   # your notebook logic moved here

# Load the dataset once when the app starts
batting = load_data()

# Define the UI
app_ui = ui.page_fluid(
    ui.h2("StatForecast App"),

    # Example: choose a player
    ui.input_selectize(
        "player",
        "Choose a player:",
        choices=sorted(batting["Name"].unique())
    ),

    # Output: display a table
    ui.output_table("player_stats")
)

# Define the server logic
def server(input, output, session):

    @output
    @render.table
    def player_stats():
        # Filter data for the selected player
        df = batting[batting["Name"] == input.player()]
        return df.head(10)  # show first 10 rows for now

# Create the Shiny app
app = App(app_ui, server)
