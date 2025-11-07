# EarthQuake_detection_fixed.py
import pandas as pd
import joblib, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---- Load and basic clean ----
df = pd.read_csv("Earthquake_1995-2023.csv", low_memory=False)
# normalize column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Fill categorical NaNs for common cols (if present)
for col in ("alert", "continent", "country"):
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

# Fill numeric NaNs with median
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols_all:
    df[c] = df[c].fillna(df[c].median())

# ---- Target & features ----
target = "tsunami"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in CSV. Columns: {df.columns.tolist()}")

X = df.drop(columns=[target])
y = df[target].astype(int)

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---- Column lists ----
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# ---- Preprocessing pipeline ----
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


preproc = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

# ---- Model pipeline ----
clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
pipeline = Pipeline([("preproc", preproc), ("clf", clf)])

# ---- Fit ----
pipeline.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---- Save artifact (pipeline + feature order) ----
artifact = {"model": pipeline, "feature_order": X.columns.tolist(), "num_cols": num_cols, "cat_cols": cat_cols}
joblib.dump(artifact, "Earthquake_model.pkl")
with open("feature_order_earthquake.json", "w") as f:
    json.dump(X.columns.tolist(), f)

print("Saved Earthquake_model.pkl and feature_order_earthquake.json")
