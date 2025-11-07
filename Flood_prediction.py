# flood_train_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib, json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Load ---
p = Path("flood.csv")
df = pd.read_csv(p, low_memory=False)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

target = "FloodProbability"
if target not in df.columns:
    raise ValueError(f"Target '{target}' not found. Columns: {df.columns.tolist()}")

# --- Quick target check ---
print("Target sample values (unique up to 20):", df[target].dropna().unique()[:20])

# --- Basic cleaning ---
# numeric median impute, categorical most frequent
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != target]
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "missing")

# --- Outlier winsorize (optional) ---
for c in num_cols:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[c] = df[c].clip(lower, upper)

# --- Features / target ---
X = df.drop(columns=[target])
y = df[target].astype(float)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing ---
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preproc = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

# --- Model pipeline ---
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
pipe = Pipeline([("preproc", preproc), ("rf", rf)])

# --- Train ---
print("Training RandomForestRegressor...")
pipe.fit(X_train, y_train)

# --- Eval ---
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2:   {r2:.4f}")

# --- Save artifact and feature order ---
artifact = {"model": pipe, "feature_order": X.columns.tolist(), "num_cols": num_cols, "cat_cols": cat_cols}
joblib.dump(artifact, "flood_model.pkl")
with open("feature_order_flood.json", "w") as f:
    json.dump(X.columns.tolist(), f)

print("Saved flood_model.pkl and feature_order_flood.json")
