import streamlit as st
from datetime import datetime, time
import pandas as pd

from backend.app import model, preprocess_input, ComplaintInput

st.set_page_config(
    page_title="UrbanPulse AI",
    page_icon="🏙️",
    layout="centered"
)

st.title("UrbanPulse AI")
st.subheader("Boston 311 Complaint Prediction")
st.write(
    "Enter complaint details to predict whether "
    "the complaint is likely to be resolved late."
)

with st.form("prediction_form"):

    open_date = st.date_input("Complaint opening date")
    open_time = st.time_input("Complaint opening time", value=time(10, 0))

    case_topic = st.text_input("Case topic")
    service_name = st.text_input("Service name")

    report_source = st.text_input("Report source")
    neighborhood = st.text_input("Neighborhood")
    public_works_district = st.text_input("Public works district")

    latitude = st.number_input(
        "Latitude", value=0.0, format="%.6f"
    )
    longitude = st.number_input(
        "Longitude", value=0.0, format="%.6f"
    )

    submitted = st.form_submit_button("Predict")

if submitted:

    if not report_source or not neighborhood or not public_works_district:
        st.error("Please complete all required fields.")
    else:
        try:
            complaint = ComplaintInput(
                open_date=datetime.combine(open_date, open_time),
                case_topic=case_topic or None,
                service_name=service_name or None,
                report_source=report_source,
                neighborhood=neighborhood,
                public_works_district=public_works_district,
                latitude=latitude if latitude != 0 else None,
                longitude=longitude if longitude != 0 else None
            )

            processed = preprocess_input(complaint)

            prediction = int(model.predict(processed)[0])
            probabilities = model.predict_proba(processed)[0]

            classes = list(model.classes_)
            late_probability = float(
                probabilities[classes.index(1)]
            )

            if prediction == 1:
                st.error("Prediction: Complaint likely to be LATE")
            else:
                st.success("Prediction: Complaint likely to be ON TIME")

            st.metric(
                "Late probability",
                f"{late_probability * 100:.2f}%"
            )

            st.metric(
                "On-time probability",
                f"{(1-late_probability) * 100:.2f}%"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")