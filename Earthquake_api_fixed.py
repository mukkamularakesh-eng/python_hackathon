# Earthquake_api_fixed.py
from flask import Flask, request, jsonify
import joblib, json, os, traceback
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "Earthquake_model.pkl"            # change if different
FEATURE_PATH = "feature_order_earthquake.json" # change if different

model = None
model_load_err = None
feature_order = None

def find_predictable(obj, _visited=None):
    """Recursively find and return the first object that has a .predict attribute.
       Returns the object, or None if not found."""
    if _visited is None:
        _visited = set()
    try:
        oid = id(obj)
        if oid in _visited:
            return None
        _visited.add(oid)
    except Exception:
        pass

    # direct estimator
    if hasattr(obj, "predict"):
        return obj

    # dict -> search values
    if isinstance(obj, dict):
        for v in obj.values():
            found = find_predictable(v, _visited)
            if found is not None:
                return found

    # list/tuple/set -> iterate
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            found = find_predictable(v, _visited)
            if found is not None:
                return found

    # object with __dict__ maybe contains estimator attributes (rare)
    try:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                found = find_predictable(v, _visited)
                if found is not None:
                    return found
    except Exception:
        pass

    return None

def load_assets():
    global model, model_load_err, feature_order
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not os.path.exists(FEATURE_PATH):
            raise FileNotFoundError(f"Feature-order file not found: {FEATURE_PATH}")

        bundle = joblib.load(MODEL_PATH)
        # try common keys first
        if isinstance(bundle, dict):
            for k in ("model", "pipeline", "estimator", "clf"):
                if k in bundle and hasattr(bundle[k], "predict"):
                    model = bundle[k]
                    break

        if model is None:
            # recursive search
            model = find_predictable(bundle)

        if model is None:
            raise ValueError("No estimator with .predict found inside the loaded joblib object.")

        with open(FEATURE_PATH, "r", encoding="utf-8") as f:
            feature_order = json.load(f)

    except Exception as e:
        model = None
        model_load_err = traceback.format_exc()
        app.logger.error("Failed to load assets:\n%s", model_load_err)

# load at startup
load_assets()

@app.route("/", methods=["GET"])
def root():
    return {
        "service": "Earthquake API",
        "status": "ok" if model is not None else "model_missing",
        "endpoints": ["/predict_earthquake (POST)"]
    }

@app.route("/predict_earthquake", methods=["POST"])
def predict_earthquake():
    if model is None:
        return jsonify({"error": "model not available", "details": model_load_err}), 500

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "invalid or empty JSON body"}), 400

    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object (or {\"data\": {...}})"}), 400

    if feature_order is None:
        return jsonify({"error": "feature_order not loaded"}), 500

    try:
        row = pd.DataFrame([{k: data.get(k, None) for k in feature_order}], columns=feature_order)
        pred = model.predict(row)

        if hasattr(pred, "tolist"):
            return jsonify({"prediction": pred.tolist()})
        else:
            return jsonify({"prediction": float(pred)})

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Prediction failed:\n%s", tb)
        return jsonify({"error": "prediction failed", "details": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
  