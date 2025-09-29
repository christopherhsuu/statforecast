from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

app = Flask(__name__)

# Example minimal model (replace with your real one)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    player = data.get("player", "Unknown")
    return jsonify({
        "player": player,
        "projection": {"HR": 30, "OPS": 0.9}  # dummy response for testing
    })

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
