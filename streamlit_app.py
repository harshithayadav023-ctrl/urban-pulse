
import streamlit as st
from datetime import datetime, time
from pathlib import Path
import pandas as pd

from backend.app import model, preprocess_input, ComplaintInput


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="UrbanPulse AI",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ UrbanPulse AI")
st.caption(
    "Boston 311 Complaint Intelligence | "
    "Prediction • Clustering • Prioritization"
)


# --------------------------------------------------
# LOAD CLUSTERING AND PRIORITIZATION RESULTS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    PROJECT_DIR
    / "outputs"
    / "clustering_prioritization_results.csv"
)

@st.cache_data
def load_results(path):
    return pd.read_csv(path)

try:
    results = load_results(CSV_PATH)
except FileNotFoundError:
    st.error(
        "Clustering results CSV not found. "
        "Please check outputs/clustering_prioritization_results.csv"
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load clustering results: {e}")
    st.stop()


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

prediction_tab, clustering_tab, priority_tab = st.tabs(
    [
        "🔮 Complaint Prediction",
        "📊 K-Means Clustering",
        "🚨 Complaint Prioritization"
    ]
)


# ==================================================
# SECTION 1: PREDICTION
# ==================================================

with prediction_tab:

    st.header("Complaint Resolution Prediction")

    st.write(
        "Enter complaint details to predict whether "
        "a complaint is likely to be resolved late or on time."
    )

    with st.form("prediction_form"):

        open_date = st.date_input("Complaint opening date")

        open_time = st.time_input(
            "Complaint opening time",
            value=time(10, 0)
        )

        case_topic = st.text_input("Case topic")
        service_name = st.text_input("Service name")

        report_source = st.text_input("Report source")
        neighborhood = st.text_input("Neighborhood")
        public_works_district = st.text_input(
            "Public works district"
        )

        latitude = st.number_input(
            "Latitude",
            value=0.0,
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            value=0.0,
            format="%.6f"
        )

        submitted = st.form_submit_button("Predict")

    if submitted:

        if (
            not report_source
            or not neighborhood
            or not public_works_district
        ):
            st.error("Please complete all required fields.")

        else:
            try:

                complaint = ComplaintInput(
                    open_date=datetime.combine(
                        open_date,
                        open_time
                    ),
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

                on_time_probability = 1 - late_probability

                st.subheader("Prediction Results")

                if prediction == 1:
                    st.error(
                        "Prediction: Complaint likely to be LATE"
                    )
                else:
                    st.success(
                        "Prediction: Complaint likely to be ON TIME"
                    )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Late Probability",
                    f"{late_probability * 100:.2f}%"
                )

                col2.metric(
                    "On-time Probability",
                    f"{on_time_probability * 100:.2f}%"
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ==================================================
# SECTION 2: K-MEANS CLUSTERING
# ==================================================

with clustering_tab:

    st.header("K-Means Complaint Clustering")

    st.write(
        "Explore the complaint groups identified by "
        "the K-means clustering workflow."
    )

    cluster_summary = (
        results.groupby("cluster")
        .agg(
            neighborhood_department_groups=(
                "cluster", "size"
            ),
            total_complaints=(
                "complaint_count", "sum"
            ),
            average_overdue_rate=(
                "overdue_rate", "mean"
            ),
            high_risk_groups=(
                "high_risk_cluster", "sum"
            )
        )
        .reset_index()
        .sort_values("cluster")
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Number of Clusters",
        results["cluster"].nunique()
    )

    col2.metric(
        "Complaint Groups",
        len(results)
    )

    col3.metric(
        "Complaints Represented",
        f"{int(results['complaint_count'].sum()):,}"
    )

    st.subheader("Cluster Distribution")

    st.bar_chart(
        cluster_summary.set_index("cluster")[
            "total_complaints"
        ]
    )

    st.subheader("Cluster Characteristics")

    display_clusters = cluster_summary.copy()

    display_clusters["average_overdue_rate"] = (
        display_clusters["average_overdue_rate"] * 100
    ).round(2)

    display_clusters = display_clusters.rename(
        columns={
            "cluster": "Cluster",
            "neighborhood_department_groups": "Neighborhood-Department Groups",
            "total_complaints": "Total Complaints",
            "average_overdue_rate": "Average Overdue Rate (%)",
            "high_risk_groups": "High-Risk Groups"
        }
    )

    st.dataframe(
        display_clusters,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Explore Individual Clusters")

    selected_cluster = st.selectbox(
        "Choose a cluster",
        sorted(results["cluster"].dropna().unique())
    )

    cluster_data = results[
        results["cluster"] == selected_cluster
    ].copy()

    st.write(
        f"Showing neighborhood and department groups "
        f"belonging to Cluster {selected_cluster}."
    )

    st.dataframe(
        cluster_data[
            [
                "neighborhood",
                "assigned_department",
                "complaint_count",
                "overdue_rate",
                "median_resolution_days",
                "high_risk_cluster"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Cluster numbers are algorithm-generated labels. "
        "They identify groups and do not, by themselves, "
        "indicate a severity ranking."
    )


# ==================================================
# SECTION 3: PRIORITIZATION
# ==================================================

with priority_tab:

    st.header("Complaint Prioritization")

    st.write(
        "Identify neighborhood and department combinations "
        "that should receive attention based on the "
        "precomputed priority scores."
    )

    priority_data = results.sort_values(
        "priority_rank"
    ).copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Groups Prioritized",
        len(priority_data)
    )

    col2.metric(
        "High-Risk Groups",
        int(priority_data["high_risk_cluster"].sum())
    )

    col3.metric(
        "Highest Priority Score",
        f"{priority_data['priority_score'].max():.3f}"
    )

    st.subheader("Priority Ranking")

    top_n = st.slider(
        "Number of priority groups to display",
        min_value=5,
        max_value=min(100, len(priority_data)),
        value=min(20, len(priority_data))
    )

    top_priority = priority_data.head(top_n)

    st.dataframe(
        top_priority[
            [
                "priority_rank",
                "neighborhood",
                "assigned_department",
                "cluster",
                "complaint_count",
                "overdue_rate",
                "open_overdue_backlog",
                "priority_score",
                "high_risk_cluster"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Filter Priorities")

    neighborhoods = sorted(
        priority_data["neighborhood"].dropna().unique()
    )

    selected_neighborhood = st.selectbox(
        "Select neighborhood",
        ["All Neighborhoods"] + neighborhoods
    )

    filtered_data = priority_data.copy()

    if selected_neighborhood != "All Neighborhoods":
        filtered_data = filtered_data[
            filtered_data["neighborhood"]
            == selected_neighborhood
        ]

    departments = sorted(
        filtered_data["assigned_department"]
        .dropna()
        .unique()
    )

    selected_department = st.selectbox(
        "Select department",
        ["All Departments"] + departments
    )

    if selected_department != "All Departments":
        filtered_data = filtered_data[
            filtered_data["assigned_department"]
            == selected_department
        ]

    st.dataframe(
        filtered_data[
            [
                "priority_rank",
                "neighborhood",
                "assigned_department",
                "complaint_count",
                "overdue_rate",
                "open_overdue_backlog",
                "priority_score",
                "high_risk_cluster"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Priority Score Comparison")

    chart_data = (
        top_priority
        .sort_values("priority_score", ascending=True)
        .copy()
    )

    chart_data["group"] = (
        chart_data["neighborhood"]
        + " - "
        + chart_data["assigned_department"]
    )

    st.bar_chart(
        chart_data.set_index("group")["priority_score"]
    )

    st.caption(
        "Higher priority scores indicate higher priority "
        "according to the project's existing prioritization "
        "calculation. Rankings are loaded from the saved results."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "UrbanPulse AI | Boston 311 Service Request Analysis | "
    "Prediction • K-Means Clustering • Prioritization"
)
