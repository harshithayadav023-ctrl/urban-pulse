import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------
# 1. PROJECT FILE LOCATIONS
# --------------------------------------------------
# On your laptop the folder is found automatically.
# On Kaggle/Colab set the environment variable URBAN_DIR to the folder that
# contains models/, common_data_corrected/ and outputs/.

PROJECT_DIR = Path(
    os.getenv("URBAN_DIR", str(Path(__file__).resolve().parent.parent))
)

MODEL_PATH = PROJECT_DIR / "models" / "harshitha_random_forest.pkl"
PREP_PATH = PROJECT_DIR / "common_data_corrected" / "prep_info.pkl"
PRIORITY_PATH = PROJECT_DIR / "outputs" / "clustering_prioritization_results.csv"

# Late / on-time cut-off. 0.5 is the default; use the threshold you chose
# from your threshold table (for example 0.45).
THRESHOLD = 0.5

# --------------------------------------------------
# 2. LOAD MODEL AND PREPROCESSING INFORMATION
# --------------------------------------------------

# Download model from Hugging Face if it is not available locally
if MODEL_PATH.exists():
    model_file = MODEL_PATH
else:
    try:
        model_file = hf_hub_download(
            repo_id="Sdc2026/urbanpulse-random-forest",
            filename="harshitha_random_forest.pkl",
            token=os.getenv("HF_TOKEN") or None
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not download model from Hugging Face: {error}"
        )

# Check preprocessing file
if not PREP_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessing file not found: {PREP_PATH}"
    )

# Load model and preprocessing information
model = joblib.load(model_file)
prep_info = joblib.load(PREP_PATH)

num_cols = prep_info["num_cols"]
cat_cols = prep_info["cat_cols"]
feature_cols = prep_info["feature_cols"]
fill_values = prep_info["fill_values"]
keep_categories = prep_info["keep_categories"]
model_columns = prep_info["columns"]

# The inputs a user can actually give (the rest are derived from them)
USER_FIELDS = ["case_topic", "report_source", "neighborhood",
               "public_works_district", "latitude", "longitude", "open_date"]

# Boston area, used in training to remove wrong coordinates
LAT_RANGE = (42.2, 42.45)
LON_RANGE = (-71.2, -70.99)

# --------------------------------------------------
# 3. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="UrbanPulse AI API",
    description="API for predicting late Boston 311 complaints",
    version="1.1.0"
)

# Needed so the dashboard (HTML/JS) can call this API with fetch()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 4. DEFINE INPUT FORMAT  (every field is optional)
# --------------------------------------------------

class ComplaintInput(BaseModel):
    open_date: Optional[datetime] = None          # default = now

    case_topic: Optional[str] = None
    service_name: Optional[str] = None
    report_source: Optional[str] = None
    neighborhood: Optional[str] = None
    public_works_district: Optional[Union[str, int]] = None   # "5" or 5 both work

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    model_config = ConfigDict(
    json_schema_extra={
        "examples": [
            {
                "open_date": "2026-11-20T10:30:00",
                "case_topic": "Pothole",
                "report_source": "BOS311",
                "neighborhood": "Allston",
                "public_works_district": "1A",
                "latitude": 42.36,
                "longitude": -71.06
            }
        ]
    }
)


def to_dict(model_obj):
    # works with pydantic v2 (model_dump) and v1 (dict)
    return model_obj.model_dump() if hasattr(model_obj, "model_dump") else model_obj.dict()

# --------------------------------------------------
# 5. PREPROCESS INCOMING COMPLAINT
# --------------------------------------------------

def preprocess_input(data: ComplaintInput):

    complaint = to_dict(data)

    # Use service_name if case_topic is not provided,
    # matching the original preprocessing notebook.
    if not complaint.get("case_topic"):
        complaint["case_topic"] = complaint.get("service_name")

    # Which inputs did the user really give? (used for the reliability note)
    missing = [f for f in USER_FIELDS
               if f != "open_date" and complaint.get(f) in (None, "")]

    # Clean values the same way as in training
    if complaint.get("neighborhood") == "Boston":
        complaint["neighborhood"] = None
        if "neighborhood" not in missing:
            missing.append("neighborhood")
    if complaint.get("public_works_district") is not None:
        complaint["public_works_district"] = str(complaint["public_works_district"])
    lat, lon = complaint.get("latitude"), complaint.get("longitude")
    if lat is not None and lon is not None:
        if not (LAT_RANGE[0] <= lat <= LAT_RANGE[1] and LON_RANGE[0] <= lon <= LON_RANGE[1]):
            complaint["latitude"] = complaint["longitude"] = None   # outside Boston -> filled later
            missing += [f for f in ("latitude", "longitude") if f not in missing]

    # Parse date (default = now)
    raw_date = complaint.get("open_date") or datetime.utcnow()
    open_date = pd.to_datetime(raw_date, errors="coerce", utc=True)

    if pd.isna(open_date):
        raise ValueError("Invalid open_date")

    # Create the date-derived features used during training
    hour = open_date.hour
    weekday = open_date.dayofweek

    complaint["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    complaint["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    complaint["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    complaint["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    complaint["is_weekend"] = int(weekday >= 5)

    # Keep only the original model input features
    row = pd.DataFrame([{col: complaint.get(col) for col in feature_cols}])

    # Convert numerical columns and fill missing values
    for col in num_cols:
        row[col] = pd.to_numeric(row[col], errors="coerce")
        row[col] = row[col].fillna(fill_values[col])

    # Fill missing categorical values
    notes = []
    for col in cat_cols:
        row[col] = row[col].fillna("Unknown").astype(str).str.strip()

        # Match the rare-category handling during training
        allowed = keep_categories[col]

        if row.loc[0, col] not in allowed:
            if row.loc[0, col] != "Unknown":
                notes.append(f"{col} '{row.loc[0, col]}' was not in the training data, so it was treated as 'Other'")
            row.loc[0, col] = "Other"

    # One-hot encode categorical columns
    row = pd.get_dummies(row, columns=cat_cols, dtype=np.int8)

    # Match the exact training feature columns and order
    row = row.reindex(columns=model_columns, fill_value=0)

    return row, missing, notes

# --------------------------------------------------
# 6. HOME / HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to UrbanPulse AI API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/options")
def options():
    # Values the model knows. Use these to fill the dropdowns in the dashboard.
    return {col: sorted(keep_categories[col]) for col in cat_cols}

# --------------------------------------------------
# 7. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_complaint(data: ComplaintInput):

    try:
        processed_data, missing, notes = preprocess_input(data)

        probability = model.predict_proba(processed_data)[0]

        classes = list(model.classes_)
        late_index = classes.index(1)

        late_probability = float(probability[late_index])
        prediction = int(late_probability >= THRESHOLD)

        result = "Late" if prediction == 1 else "On time"

        # simple reliability note: fewer inputs -> less reliable
        given = 6 - len([m for m in missing if m != "open_date"])
        reliability = "High" if given >= 5 else "Medium" if given >= 3 else "Low"

        return {
            "prediction": prediction,
            "result": result,
            "late_probability": round(late_probability, 4),
            "on_time_probability": round(1 - late_probability, 4),
            "threshold": THRESHOLD,
            "missing_fields": missing,
            "inputs_given": f"{given} of 6",
            "reliability": reliability,
            "notes": notes,
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

# --------------------------------------------------
# 8. PRIORITY ENDPOINT
# --------------------------------------------------

@app.get("/priorities")
def get_priorities(limit: int = 10):

    if not PRIORITY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Prioritisation results file not found"
        )

    priority_df = pd.read_csv(PRIORITY_PATH)

    if limit < 1 or limit > len(priority_df):
        raise HTTPException(
            status_code=400,
            detail=f"Limit must be between 1 and {len(priority_df)}"
        )

    result = priority_df.head(limit)

    # empty (NaN) cells cannot be sent as JSON, so convert them to null
    result = result.astype(object).where(pd.notna(result), None)

    return result.to_dict(orient="records")