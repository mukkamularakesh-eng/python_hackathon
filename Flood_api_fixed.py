# flood_api_fixed.py
from flask import Flask, request, jsonify
import joblib, json, os, traceback
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "flood_model.pkl"            # change if different
FEATURE_PATH = "feature_order_flood.json" # change if different

model = None
model_load_err = None
feature_order = None

# reuse the same recursive finder
def find_predictable(obj, _visited=None):
    if _visited is None:
        _visited = set()
    try:
        oid = id(obj)
        if oid in _visited:
            return None
        _visited.add(oid)
    except Exception:
        pass

    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for v in obj.values():
            found = find_predictable(v, _visited)
            if found is not None:
                return found

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            found = find_predictable(v, _visited)
            if found is not None:
                return found

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
        if isinstance(bundle, dict):
            for k in ("model", "pipeline", "estimator", "clf"):
                if k in bundle and hasattr(bundle[k], "predict"):
                    model = bundle[k]
                    break

        if model is None:
            model = find_predictable(bundle)

        if model is None:
            raise ValueError("No estimator with .predict found inside the loaded joblib object.")

        with open(FEATURE_PATH, "r", encoding="utf-8") as f:
            feature_order = json.load(f)

    except Exception as e:
        model = None
        model_load_err = traceback.format_exc()
        app.logger.error("Failed to load assets:\n%s", model_load_err)

load_assets()

@app.before_request
def normalize_path_trailing_whitespace():
    """
    If the incoming path contains accidental trailing whitespace (e.g. /predict_flood%20),
    silently rewrite PATH_INFO in the WSGI environ to strip trailing whitespace.
    This lets Flask route the request to the correct endpoint without returning 404
    or issuing a redirect. It preserves method and body.
    """
    # request.path is the decoded path (so '%20' becomes a space)
    path = request.path
    stripped = path.rstrip()
    if path != stripped:
        # mutate WSGI environ so routing will see the cleaned path
        # preserve query string (PATH_INFO should not include query)
        request.environ['PATH_INFO'] = stripped
        app.logger.debug("Normalized path: %r -> %r", path, stripped)
        # continue handling the (rewritten) request

@app.route("/", methods=["GET"])
def root():
    return {
        "service": "Flood API",
        "status": "ok" if model is not None else "model_missing",
        "endpoints": ["/predict_flood (POST)"]
    }

# add GET so browser tests don't produce 405
@app.route("/predict_flood", methods=["GET", "POST"])
def predict_flood():
    if request.method == "GET":
        example = {
            "data": {k: "<value>" for k in (feature_order if isinstance(feature_order, list) else [])}
        }
        return (
            "<h1>Flood API</h1>"
            "<p>POST JSON to <code>/predict_flood</code> with either a raw dict of features or {'data': {...}}.</p>"
            "<p>Example curl:</p>"
            f"<pre>curl -X POST http://localhost:5000/predict_flood -H 'Content-Type: application/json' -d '{json.dumps(example)}'</pre>",
            200,
            {"Content-Type": "text/html"}
        )

    # POST handling
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

    missing = [f for f in feature_order if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400

    try:
        row = pd.DataFrame([{k: data.get(k, None) for k in feature_order}], columns=feature_order)
        pred = model.predict(row)

        if hasattr(pred, "tolist"):
            return jsonify({"flood_prediction": pred.tolist()})
        else:
            # handle numpy scalars too
            try:
                val = float(pred)
            except Exception:
                val = pred
            return jsonify({"flood_prediction": val})

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Prediction failed:\n%s", tb)
        return jsonify({"error": "prediction failed", "details": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    # set debug=False in production
    app.run(debug=True, host="0.0.0.0", port=5000)
