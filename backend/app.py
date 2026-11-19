
from pathlib import Path
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------
# 1. PROJECT FILE LOCATIONS
# --------------------------------------------------
from pathlib import Path
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_FILENAME = "harshitha_random_forest.pkl"
HF_REPO_ID = "Sdc2026/urbanpulse-random-forest"

LOCAL_MODEL_PATH = PROJECT_DIR / "models" / MODEL_FILENAME
PREP_PATH = PROJECT_DIR / "common_data_corrected" / "prep_info.pkl"
PRIORITY_PATH = PROJECT_DIR / "outputs" / "clustering_prioritization_results.csv"

# --------------------------------------------------
# 2. LOAD MODEL AND PREPROCESSING INFORMATION
# --------------------------------------------------
if LOCAL_MODEL_PATH.exists():
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
    )

if not PREP_PATH.exists():
    raise FileNotFoundError(f"Preprocessing file not found: {PREP_PATH}")

model = joblib.load(MODEL_PATH)
prep_info = joblib.load(PREP_PATH)

num_cols = prep_info["num_cols"]
cat_cols = prep_info["cat_cols"]
feature_cols = prep_info["feature_cols"]
fill_values = prep_info["fill_values"]
keep_categories = prep_info["keep_categories"]
model_columns = prep_info["columns"]



# --------------------------------------------------
# 3. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="UrbanPulse AI API",
    description="API for predicting late Boston 311 complaints",
    version="1.0.0"
)

# --------------------------------------------------
# 4. DEFINE INPUT FORMAT
# --------------------------------------------------

class ComplaintInput(BaseModel):
    open_date: datetime

    case_topic: Optional[str] = None
    service_name: Optional[str] = None
    report_source: str
    neighborhood: str
    public_works_district: str

    latitude: Optional[float] = None
    longitude: Optional[float] = None

# --------------------------------------------------
# 5. PREPROCESS INCOMING COMPLAINT
# --------------------------------------------------

def preprocess_input(data: ComplaintInput):

    complaint = data.model_dump()

    # Use service_name if case_topic is not provided,
    # matching the original preprocessing notebook.
    if not complaint.get("case_topic"):
        complaint["case_topic"] = complaint.get("service_name")

    # Parse date
    open_date = pd.to_datetime(
        complaint["open_date"],
        errors="coerce",
        utc=True
    )

    if pd.isna(open_date):
        raise ValueError("Invalid open_date")

    # Create the date-derived features used during training
    hour = open_date.hour
    weekday = open_date.dayofweek

    complaint["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    complaint["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    complaint["weekday_sin"] = np.sin(
        2 * np.pi * weekday / 7
    )
    complaint["weekday_cos"] = np.cos(
        2 * np.pi * weekday / 7
    )

    complaint["is_weekend"] = int(weekday >= 5)

    # Keep only the original model input features
    row = pd.DataFrame([{
        col: complaint.get(col)
        for col in feature_cols
    }])

    # Convert numerical columns and fill missing values
    for col in num_cols:
        row[col] = pd.to_numeric(
            row[col], errors="coerce"
        )

        row[col] = row[col].fillna(
            fill_values[col]
        )

    
    # Fill missing categorical values and match categories
    # without considering uppercase/lowercase differences.
    for col in cat_cols:

        row[col] = (
            row[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        # Categories saved during training
        allowed = keep_categories[col]

        # Create a case-insensitive lookup while preserving
        # the original spelling used during model training.
        category_lookup = {
            str(category).strip().casefold(): category
            for category in allowed
        }

        input_value = row.loc[0, col]

        # Match the input to its original training category
        matched_category = category_lookup.get(
            str(input_value).strip().casefold()
        )

        if matched_category is not None:
            row.loc[0, col] = matched_category
        else:
            row.loc[0, col] = "Other"

    # One-hot encode categorical columns
    row = pd.get_dummies(
        row,
        columns=cat_cols,
        dtype=np.int8
    )

    # Match the exact training feature columns and order
    row = row.reindex(
        columns=model_columns,
        fill_value=0
    )

    return row

# --------------------------------------------------
# 6. HOME / HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to UrbanPulse AI API",
        "status": "running"
    }

# --------------------------------------------------
# 7. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_complaint(data: ComplaintInput):

    try:
        processed_data = preprocess_input(data)

        prediction = int(model.predict(processed_data)[0])

        probability = model.predict_proba(
            processed_data
        )[0]

        classes = list(model.classes_)
        late_index = classes.index(1)

        late_probability = float(probability[late_index])

        result = (
            "Late"
            if prediction == 1
            else "On time"
        )

        return {
            "prediction": prediction,
            "result": result,
            "late_probability": round(
                late_probability, 4
            ),
            "on_time_probability": round(
                1 - late_probability, 4
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

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

    if limit < 1 or limit > 218:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 218"
        )

    priority_df = pd.read_csv(PRIORITY_PATH)

    result = priority_df.head(limit)

    return result.to_dict(orient="records")