import streamlit as st
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from backend.app import model, preprocess_input, ComplaintInput, prep_info

# --------------------------------------------------
# PAGE SETUP AND DATA
# --------------------------------------------------
st.set_page_config(page_title="UrbanPulse AI", page_icon="🏙️", layout="wide")
st.title("🏙️ UrbanPulse AI")
st.caption("Boston 311 Complaint Intelligence | Prediction • Clustering • Prioritization")

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "outputs" / "clustering_prioritization_results.csv"

@st.cache_data
def load_results(path):
    return pd.read_csv(path)

try:
    results = load_results(CSV_PATH)
except FileNotFoundError:
    st.error("Clustering results CSV not found: outputs/clustering_prioritization_results.csv")
    st.stop()
except Exception as exc:
    st.error(f"Could not load clustering results: {exc}")
    st.stop()

# Read the exact categories retained during model preprocessing.
KEEP_CATEGORIES = prep_info.get("keep_categories", {})


def category_options(field, fallback=None):
    """Return training-time categories; use results CSV only as a safe fallback."""
    values = KEEP_CATEGORIES.get(field, [])
    if values is None:
        values = []
    values = [str(v) for v in values if pd.notna(v) and str(v).strip()]
    if not values and fallback is not None:
        values = [str(v) for v in fallback if pd.notna(v) and str(v).strip()]
    # Keep the model's original category spellings; remove duplicates only.
    return list(dict.fromkeys(values))


def show_columns(frame, wanted):
    """Select only columns that really exist in the supplied results file."""
    return [column for column in wanted if column in frame.columns]


def neighborhood_coordinates(neighborhood):
    """Use the selected neighborhood's median coordinates when available."""
    if {"neighborhood", "latitude", "longitude"}.issubset(results.columns):
        area_rows = results[
            results["neighborhood"].astype(str).str.casefold()
            == str(neighborhood).casefold()
        ].copy()
        area_rows["latitude"] = pd.to_numeric(area_rows["latitude"], errors="coerce")
        area_rows["longitude"] = pd.to_numeric(area_rows["longitude"], errors="coerce")
        area_rows = area_rows.dropna(subset=["latitude", "longitude"])
        if not area_rows.empty:
            return float(area_rows["latitude"].median()), float(area_rows["longitude"].median()), "selected neighborhood"

    # Fallback matches the training-time imputation rule stored with the model.
    fill_values = prep_info.get("fill_values", {})
    lat = fill_values.get("latitude")
    lon = fill_values.get("longitude")
    if lat is not None and lon is not None:
        return float(lat), float(lon), "training-data median (neighborhood coordinates were not available)"
    return None, None, "no saved coordinate fallback was found"


# --------------------------------------------------
# TABS
# --------------------------------------------------
prediction_tab, clustering_tab, priority_tab = st.tabs([
    "🔮 Complaint Prediction",
    "📊 K-Means Clustering",
    "🚨 Complaint Prioritization",
])

# ==================================================
# 1. COMPLAINT PREDICTION
# ==================================================
with prediction_tab:
    st.header("Complaint Resolution Prediction")
    st.write("Enter complaint details to estimate whether it is likely to be resolved late or on time.")

    # Keep this checkbox OUTSIDE the form. Streamlit reruns immediately when
    # it changes, so the coordinate fields appear/disappear before Predict.
    add_coordinates = st.checkbox(
        "I want to enter latitude and longitude myself",
        value=False,
        key="enter_coordinates_checkbox",
        help="Leave unticked to use the selected neighborhood's saved median coordinates."
    )

    with st.form("prediction_form"):
        open_date = st.date_input("Complaint opening date")
        open_time = st.time_input("Complaint opening time", value=time(10, 0))

        # Use the same categorical feature names as the Kaggle-trained model.
        # In this model, complaint type is case_topic, not service_name.
        service_options = category_options("case_topic")
        if not service_options:
            st.error("No case_topic categories were found in prep_info.pkl. Cannot build the service-type dropdown.")
            st.stop()
        case_topic = st.selectbox("Service type / complaint type", service_options, key="prediction_case_topic")

        report_options = category_options("report_source")
        if not report_options:
            st.error("No report_source categories were found in prep_info.pkl.")
            st.stop()
        report_source = st.selectbox("Report source", report_options, key="prediction_report_source")

        neighborhood_fallback = results["neighborhood"].dropna().unique() if "neighborhood" in results.columns else None
        neighborhood_options = category_options("neighborhood", neighborhood_fallback)
        if not neighborhood_options:
            st.error("No neighborhood categories were found in prep_info.pkl or the clustering results CSV.")
            st.stop()
        neighborhood = st.selectbox("Neighborhood", neighborhood_options, key="prediction_neighborhood")

        district_fallback = results["public_works_district"].dropna().unique() if "public_works_district" in results.columns else None
        district_options = category_options("public_works_district", district_fallback)
        if not district_options:
            st.error("No public works district categories were found in prep_info.pkl or the clustering results CSV.")
            st.stop()
        public_works_district = st.selectbox("Public Works District", district_options, key="prediction_district")

        if add_coordinates:
            coord_col1, coord_col2 = st.columns(2)
            latitude = coord_col1.number_input(
                "Latitude (Boston area)", min_value=42.20, max_value=42.40,
                value=42.36, step=0.0001, format="%.6f",
                help="Approximate Boston-area bounds.",
            )
            longitude = coord_col2.number_input(
                "Longitude (Boston area)", min_value=-71.20, max_value=-70.90,
                value=-71.06, step=0.0001, format="%.6f",
                help="Approximate Boston-area bounds.",
            )
        else:
            latitude = None
            longitude = None

        submitted = st.form_submit_button("Predict")

    if submitted:
        if not case_topic or not report_source or not neighborhood or not public_works_district:
            st.error("Please complete all required fields.")
        else:
            try:
                if not add_coordinates:
                    latitude, longitude, coordinate_source = neighborhood_coordinates(neighborhood)
                    if latitude is None or longitude is None:
                        st.error(
                            "Coordinates are optional, but no saved coordinates or training median were found. "
                            "Please tick the coordinate checkbox and enter them, or provide a source file "
                            "containing neighborhood, latitude and longitude."
                        )
                        st.stop()
                    st.info(f"Using coordinates from: {coordinate_source}.")

                complaint = ComplaintInput(
                    open_date=datetime.combine(open_date, open_time),
                    case_topic=case_topic,
                    service_name=None,
                    report_source=report_source,
                    neighborhood=neighborhood,
                    public_works_district=public_works_district,
                    latitude=latitude,
                    longitude=longitude,
                )
                processed = preprocess_input(complaint)
                prediction = int(model.predict(processed)[0])
                probabilities = model.predict_proba(processed)[0]
                classes = list(model.classes_)
                late_probability = float(probabilities[classes.index(1)])
                on_time_probability = 1.0 - late_probability

                st.subheader("Prediction Results")
                if prediction == 1:
                    st.error("Prediction: Complaint likely to be LATE")
                else:
                    st.success("Prediction: Complaint likely to be ON TIME")

                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Late Probability", f"{late_probability * 100:.2f}%")
                result_col2.metric("On-time Probability", f"{on_time_probability * 100:.2f}%")
            except Exception as exc:
                st.error(f"Prediction error: {exc}")

# ==================================================
# 2. K-MEANS CLUSTERING
# ==================================================
with clustering_tab:
    st.header("K-Means Complaint Clustering")
    st.write("K-Means groups similar neighborhood-and-department complaint patterns. Each row below is a group within a cluster.")

    required_cluster_cols = {"cluster", "complaint_count", "overdue_rate", "high_risk_cluster"}
    if not required_cluster_cols.issubset(results.columns):
        st.error("The clustering results file is missing one or more required columns: " + ", ".join(sorted(required_cluster_cols - set(results.columns))))
    else:
        cluster_summary = (
            results.groupby("cluster", dropna=False)
            .agg(
                neighborhood_department_groups=("cluster", "size"),
                total_complaints=("complaint_count", "sum"),
                average_overdue_rate=("overdue_rate", "mean"),
                high_risk_groups=("high_risk_cluster", "sum"),
            )
            .reset_index()
            .sort_values("cluster")
        )

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Clusters", results["cluster"].nunique())
        metric2.metric("Neighborhood–department groups", len(results))
        metric3.metric("Complaints represented", f"{int(results['complaint_count'].sum()):,}")

        st.subheader("Complaints in Each Cluster")
        st.bar_chart(cluster_summary.set_index("cluster")["total_complaints"])

        display_summary = cluster_summary.copy()
        display_summary["average_overdue_rate"] = (display_summary["average_overdue_rate"] * 100).round(2)
        display_summary = display_summary.rename(columns={
            "cluster": "Cluster",
            "neighborhood_department_groups": "Area–department groups",
            "total_complaints": "Total complaints",
            "average_overdue_rate": "Average overdue rate (%)",
            "high_risk_groups": "High-risk groups",
        })
        st.subheader("Cluster Summary")
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

        st.subheader("What is inside a cluster?")
        cluster_values = sorted(results["cluster"].dropna().unique().tolist())
        if cluster_values:
            selected_cluster = st.selectbox("Choose a cluster", cluster_values, key="cluster_detail_select")
            cluster_data = results[results["cluster"] == selected_cluster].copy()
            st.caption(f"Cluster {selected_cluster}: neighborhood and assigned-department groups with their complaint volume and overdue indicators.")
            detail_cols = show_columns(cluster_data, [
                "neighborhood", "assigned_department", "complaint_count",
                "overdue_rate", "median_resolution_days", "open_overdue_backlog",
                "high_risk_cluster", "priority_rank", "priority_score",
            ])
            st.dataframe(cluster_data[detail_cols], use_container_width=True, hide_index=True)
        st.caption("Cluster IDs are labels created by K-Means; a larger cluster number does not mean higher priority.")

# ==================================================
# 3. COMPLAINT PRIORITIZATION
# ==================================================
with priority_tab:
    st.header("Complaint Prioritization")
    st.write("Each neighborhood–department group receives a score from its complaint volume, overdue rate, open overdue backlog and high-risk-cluster flag.")

    priority_factors = [
        "complaint_count",
        "overdue_rate",
        "open_overdue_backlog",
        "high_risk_cluster",
    ]
    missing_factors = [column for column in priority_factors if column not in results.columns]

    if missing_factors:
        st.error(
            "Cannot calculate the transparent priority score because these required fields are missing: "
            + ", ".join(missing_factors)
        )
    else:
        priority_data = results.copy()

        # Convert each factor to numeric values. The high-risk flag is treated as 0/1.
        for column in priority_factors:
            priority_data[column] = pd.to_numeric(priority_data[column], errors="coerce")

        priority_data[priority_factors] = priority_data[priority_factors].fillna(0)
        priority_data["high_risk_cluster"] = priority_data["high_risk_cluster"].clip(0, 1)

        # Min-max scale every factor to 0–1. Constant columns become zero.
        scaled_columns = []
        for column in priority_factors:
            scaled_column = f"{column}_scaled"
            minimum = priority_data[column].min()
            maximum = priority_data[column].max()
            if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
                priority_data[scaled_column] = 0.0
            else:
                priority_data[scaled_column] = (priority_data[column] - minimum) / (maximum - minimum)
            scaled_columns.append(scaled_column)

        # Equal-weight baseline: 25% for each of the four available indicators.
        priority_data["priority_score"] = priority_data[scaled_columns].mean(axis=1)
        priority_data = priority_data.sort_values("priority_score", ascending=False).reset_index(drop=True)
        priority_data["priority_rank"] = range(1, len(priority_data) + 1)

        p1, p2, p3 = st.columns(3)
        p1.metric("Groups ranked", len(priority_data))
        p2.metric("High-risk groups", int(priority_data["high_risk_cluster"].sum()))
        if not priority_data.empty:
            p3.metric("Highest priority score", f"{priority_data['priority_score'].max():.3f}")

        st.subheader("Priority Ranking")
        top_n = st.slider("Groups to display", min_value=1, max_value=min(100, len(priority_data)), value=min(20, len(priority_data)))
        top_priority = priority_data.head(top_n)
        priority_cols = show_columns(top_priority, [
            "priority_rank", "neighborhood", "assigned_department", "cluster",
            "complaint_count", "overdue_rate", "open_overdue_backlog",
            "high_risk_cluster", "priority_score",
        ])
        st.dataframe(top_priority[priority_cols], use_container_width=True, hide_index=True)

        st.subheader("Filter by area or department")
        filter_col1, filter_col2 = st.columns(2)
        neighborhoods = sorted(priority_data["neighborhood"].dropna().astype(str).unique().tolist())
        selected_neighborhood = filter_col1.selectbox("Neighborhood", ["All neighborhoods"] + neighborhoods, key="priority_neighborhood_filter")
        filtered = priority_data.copy()
        if selected_neighborhood != "All neighborhoods":
            filtered = filtered[filtered["neighborhood"].astype(str) == selected_neighborhood]

        departments = sorted(filtered["assigned_department"].dropna().astype(str).unique().tolist())
        selected_department = filter_col2.selectbox("Assigned department", ["All departments"] + departments, key="priority_department_filter")
        if selected_department != "All departments":
            filtered = filtered[filtered["assigned_department"].astype(str) == selected_department]

        filtered_cols = show_columns(filtered, [
            "priority_rank", "neighborhood", "assigned_department", "cluster",
            "complaint_count", "overdue_rate", "open_overdue_backlog",
            "high_risk_cluster", "priority_score",
        ])
        st.dataframe(filtered[filtered_cols], use_container_width=True, hide_index=True)

        st.subheader("Priority Score Comparison")
        chart_data = top_priority.sort_values("priority_score", ascending=True).copy()
        chart_data["area_department"] = chart_data["neighborhood"].astype(str) + " — " + chart_data["assigned_department"].astype(str)
        st.bar_chart(chart_data.set_index("area_department")["priority_score"])

        with st.expander("How is the priority score calculated?"):
            st.write("We use a transparent equal-weight baseline. First, each indicator is min-max scaled to a value between 0 and 1. Then we take the average of the four scaled values:")
            st.code("priority_score = (complaint_count_scaled + overdue_rate_scaled + open_overdue_backlog_scaled + high_risk_cluster_scaled) / 4", language="text")
            st.markdown(
                "- **Complaint count (25%)** — higher volume means more complaints are affected.\n"
                "- **Overdue rate (25%)** — a higher share of overdue complaints indicates greater delay risk.\n"
                "- **Open overdue backlog (25%)** — more overdue complaints still open means more pending work.\n"
                "- **High-risk cluster flag (25%)** — groups marked high risk receive an additional signal."
            )
            st.caption("A higher score means the group ranks higher for review. These equal weights are an explicit project baseline, not learned model weights or an official City of Boston formula. The rank and score are recalculated from the current results whenever the app runs. Cluster IDs are not included as a numeric factor.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("UrbanPulse AI | Boston 311 Service Request Analysis | Prediction • K-Means Clustering • Prioritization")
