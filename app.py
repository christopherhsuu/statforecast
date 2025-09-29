from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

app = Flask(__name__)

# Example minimal model (replace with your real one)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    player = data["player"]
    projection = project_next_season(player, batting, model)
    return jsonify(projection)

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
