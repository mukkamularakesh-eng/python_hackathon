# Hackathon_VS - How to run APIs locally

## Prereqs
- Python 3.10+ installed
- (Windows) PowerShell

## Create & activate virtualenv
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1

## Install requirements
pip install --upgrade pip
pip install -r requirements.txt

## Files included
- Earthquake_api_fixed.py
- flood_api.py
- Pipeline.py
- Earthquake_model.pkl
- flood_model.pkl
- feature_order_earthquake.json
- feature_order_flood.json
- sample_earthquake.json
- sample_flood.json

## Run services
# Terminal 1
python Earthquake_api_fixed.py      # listens on port 5001

# Terminal 2
python flood_api.py                 # listens on port 5000

## Test endpoints (example)
# Earthquake
curl -X POST http://127.0.0.1:5001/predict_earthquake -H "Content-Type: application/json" -d @sample_earthquake.json

# Flood
curl -X POST http://127.0.0.1:5000/predict_flood -H "Content-Type: application/json" -d @sample_flood.json

## Notes
- Do NOT include .venv in zip. Backend will recreate environment.
- If model unpickling fails with ModuleNotFoundError for 'Pipeline', ensure Pipeline.py is present in the same folder.
