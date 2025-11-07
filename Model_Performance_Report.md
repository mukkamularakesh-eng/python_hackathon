# 🌍 Disaster Response and Prediction Platform
### Data Science & AI Team — Model Performance Report
_Generated on 2025-11-05 12:02:44_

## ✅ Project Overview

This project focuses on developing AI models for disaster prediction, specifically:
- **Flood Prediction** using Random Forest Regressor  
- **Earthquake Detection** using Gradient Boosting Classifier  

The models are trained on real datasets and deployed via Flask APIs for inference.

## 🌋 Earthquake Detection Model
**Model:** Gradient Boosting Classifier
**Features used:** 18
**Feature sample:** ['title', 'magnitude', 'date_time', 'cdi', 'mmi', 'alert', 'sig', 'net', 'nst', 'dmin'] ...
**Accuracy:** 0.91 (on validation set)  

**Model file:** `Earthquake_model.pkl`
**Endpoint:** `/predict_earthquake` (Flask @ port 5001)

## 🌊 Flood Prediction Model
**Model:** Random Forest Regressor
**Features used:** 20
**Feature sample:** ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices', 'Encroachments'] ...
**RMSE:** 0.0257  
**R²:** 0.7353
**Model file:** `flood_model.pkl`
**Endpoint:** `/predict_flood` (Flask @ port 5000)

## ⚙️ Flask API Integration

Both models are integrated into Flask APIs for RESTful prediction endpoints:

| Model | File | Endpoint | Port |
|--------|------|-----------|------|
| Earthquake | `Earthquake_api.py` | `/predict_earthquake` | 5001 |
| Flood | `flood_api.py` | `/predict_flood` | 5000 |

Each API supports POST requests with JSON payloads, validates inputs, and returns predictions in JSON.

## 📦 Deliverables Summary

| Deliverable | Status | File |
|--------------|---------|------|
| Flood Model | ✅ Trained | flood_model.pkl |
| Earthquake Model | ✅ Trained | Earthquake_model.pkl |
| Flood API | ✅ Done | flood_api.py |
| Earthquake API | ✅ Done | Earthquake_api.py |
| Model Report | ✅ Generated | Model_Performance_Report.md |

## 📈 Project Completion Summary

| Category | Completion |
|-----------|-------------|
| Data Collection | ✅ 100% |
| Pre-processing & EDA | ✅ 100% |
| Model Building | ✅ 100% |
| Evaluation | ✅ 100% |
| Flask Integration | ✅ 100% |
| Visualization | ✅ 80% |
| Power BI Export | 🟡 50% |
| Documentation | ✅ 90% |


✅ **Overall Completion: ~90–95%**

_Next Steps:_ Add Power BI export + optional FastAPI docs integration.
