
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"

DEFAULT_API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LLM Evaluation Agent",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# --------------------------------------------------
# API FUNCTIONS
# --------------------------------------------------

def get_api_url():
    return st.session_state.api_url.rstrip("/")


def check_api_health():

    try:
        response = requests.get(
            f"{get_api_url()}/api/v1/health",
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException:
        return None


def get_available_models():

    response = requests.get(
        f"{get_api_url()}/api/v1/models",
        timeout=10
    )

    response.raise_for_status()
    return response.json().get("available_models", [])


def run_evaluation(
    dataset_file,
    models,
    judge_model,
    enable_judge
):

    payload = {
        "dataset_file": dataset_file,
        "models": models,
        "judge_model": judge_model,
        "enable_judge": enable_judge
    }

    response = requests.post(
        f"{get_api_url()}/api/v1/evaluations",
        json=payload,
        timeout=1800
    )

    response.raise_for_status()
    return response.json()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Configuration")

st.sidebar.text_input(
    "FastAPI Base URL",
    value=DEFAULT_API_URL,
    key="api_url"
)

st.sidebar.caption(
    "The FastAPI server must be running before starting an evaluation."
)

if st.sidebar.button("Check API Connection", use_container_width=True):

    health = check_api_health()

    if health:
        st.sidebar.success("API connection successful!")
    else:
        st.sidebar.error("Cannot connect to FastAPI.")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 LLM Evaluation Agent")

st.markdown(
    """
    Evaluate, compare, and rank Large Language Models
    using automated metrics and an LLM-as-a-Judge.
    """
)

st.divider()


# --------------------------------------------------
# API STATUS
# --------------------------------------------------

health = check_api_health()

if health:

    st.success(
        f"Backend connected: {health.get('application', 'FastAPI')} "
        f"v{health.get('version', '')}"
    )

else:

    st.error(
        "FastAPI backend is not available. "
        "Start the backend before using the dashboard."
    )

    st.code(
        "python -m uvicorn backend.app.main:app --reload",
        language="powershell"
    )

    st.stop()


# --------------------------------------------------
# FETCH AVAILABLE MODELS
# --------------------------------------------------

try:

    available_models = get_available_models()

except requests.exceptions.RequestException as error:

    st.error(f"Could not retrieve models: {error}")
    st.stop()


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

overview_tab, evaluation_tab, leaderboard_tab, results_tab, reports_tab = (
    st.tabs([
        "📊 Overview",
        "🚀 Run Evaluation",
        "🏆 Leaderboard",
        "📋 Results",
        "📁 Reports"
    ])
)


# --------------------------------------------------
# OVERVIEW TAB
# --------------------------------------------------

with overview_tab:

    st.subheader("Dashboard Overview")

    last_result = st.session_state.last_result

    if last_result:

        summary = last_result.get("summary", {})
        ranking = last_result.get("ranking", [])

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Evaluations",
            summary.get("total_questions", 0)
        )

        col2.metric(
            "Successful",
            summary.get("successful_questions", 0)
        )

        col3.metric(
            "Failed",
            summary.get("failed_questions", 0)
        )

        col4.metric(
            "Models Compared",
            len(ranking)
        )

        st.subheader("Latest Model Ranking")

        if ranking:

            ranking_df = pd.DataFrame(ranking)

            display_columns = [
                "rank",
                "model",
                "quality_score",
                "avg_latency"
            ]

            st.dataframe(
                ranking_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No ranking data available.")

    else:

        st.info(
            "No evaluation has been run in this dashboard session. "
            "Open Run Evaluation to start your first benchmark."
        )

    st.subheader("Configured Models")

    st.write(available_models)


# --------------------------------------------------
# RUN EVALUATION TAB
# --------------------------------------------------

with evaluation_tab:

    st.subheader("Configure a Benchmark")

    dataset_files = sorted(
        [
            file.name
            for file in DATA_DIRECTORY.iterdir()
            if file.is_file()
            and file.suffix.lower() in [".json", ".jsonl"]
        ]
    ) if DATA_DIRECTORY.exists() else []

    if not dataset_files:

        st.warning("No JSON or JSONL datasets were found in the data folder.")

    else:

        selected_dataset = st.selectbox(
            "Select Dataset",
            dataset_files
        )

        selected_models = st.multiselect(
            "Select Models to Compare",
            options=available_models,
            default=[
                model
                for model in [
                    "llama3.2:3b",
                    "qwen2.5-coder:1.5b"
                ]
                if model in available_models
            ]
        )

        selected_judge = st.selectbox(
            "Select Judge Model",
            options=available_models,
            index=(
                available_models.index("llama3.2:3b")
                if "llama3.2:3b" in available_models
                else 0
            )
        )

        enable_judge = st.checkbox(
            "Enable LLM Judge",
            value=True
        )

        st.divider()

        st.write("**Benchmark Configuration**")

        st.write(f"Dataset: `{selected_dataset}`")
        st.write(f"Models selected: `{len(selected_models)}`")
        st.write(f"Judge: `{selected_judge}`")
        st.write(f"LLM Judge enabled: `{enable_judge}`")

        start_evaluation = st.button(
            "🚀 Start Evaluation",
            type="primary",
            use_container_width=True,
            disabled=not selected_models
        )

        if start_evaluation:

            try:

                with st.spinner(
                    "Running benchmark... This may take several minutes."
                ):

                    result = run_evaluation(
                        dataset_file=selected_dataset,
                        models=selected_models,
                        judge_model=selected_judge,
                        enable_judge=enable_judge
                    )

                st.session_state.last_result = result

                st.success("Evaluation completed successfully!")

                st.rerun()

            except requests.exceptions.HTTPError as error:

                detail = error.response.text if error.response else str(error)

                st.error(f"Evaluation API error: {detail}")

            except requests.exceptions.RequestException as error:

                st.error(f"Could not complete API request: {error}")

            except Exception as error:

                st.error(f"Unexpected error: {error}")


# --------------------------------------------------
# LEADERBOARD TAB
# --------------------------------------------------

with leaderboard_tab:

    st.subheader("Model Leaderboard")

    last_result = st.session_state.last_result

    if last_result:

        ranking = last_result.get("ranking", [])

        if ranking:

            ranking_df = pd.DataFrame(ranking)

            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True
            )

            chart_df = ranking_df[
                ["model", "quality_score"]
            ].set_index("model")

            st.subheader("Model Quality Comparison")

            st.bar_chart(chart_df)

            latency_df = ranking_df[
                ["model", "avg_latency"]
            ].set_index("model")

            st.subheader("Average Response Latency")

            st.bar_chart(latency_df)

        else:

            st.warning("No model rankings are available.")

    else:

        st.info("Run an evaluation to view the leaderboard.")


# --------------------------------------------------
# RESULTS TAB
# --------------------------------------------------

with results_tab:

    st.subheader("Individual Evaluation Results")

    last_result = st.session_state.last_result

    if last_result:

        results = last_result.get("results", [])

        if results:

            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("Inspect an Individual Answer")

            selected_index = st.selectbox(
                "Select Evaluation",
                range(len(results)),
                format_func=lambda index: (
                    f"{results[index]['model']} - "
                    f"{results[index]['sample_id']}"
                )
            )

            selected_result = results[selected_index]

            st.write("**Question**")
            st.write(selected_result.get("question"))

            st.write("**Expected Answer**")
            st.write(selected_result.get("expected_answer"))

            st.write("**Generated Answer**")
            st.write(selected_result.get("predicted_answer"))

            st.write("**Evaluation Metrics**")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Exact Match",
                selected_result.get("exact_match")
                if selected_result.get("exact_match") is not None
                else "N/A"
            )

            metric2.metric(
                "Token F1",
                selected_result.get("token_f1")
                if selected_result.get("token_f1") is not None
                else "N/A"
            )

            metric3.metric(
                "Semantic Similarity",
                selected_result.get("semantic_similarity")
                if selected_result.get("semantic_similarity") is not None
                else "N/A"
            )

            st.write("**LLM Judge Reasoning**")
            st.write(
                selected_result.get("judge_reasoning")
                or "Judge evaluation unavailable."
            )

        else:

            st.info("No individual evaluation results available.")

    else:

        st.info("Run an evaluation to inspect individual results.")


# --------------------------------------------------
# REPORTS TAB
# --------------------------------------------------

with reports_tab:

    st.subheader("Generated Evaluation Reports")

    last_result = st.session_state.last_result

    if last_result:

        report_paths = last_result.get("reports", {})

        for report_type, label in [
            ("json_report", "JSON Report"),
            ("csv_report", "CSV Report")
        ]:

            report_path = report_paths.get(report_type)

            if report_path and Path(report_path).is_file():

                file_path = Path(report_path)

                with file_path.open("rb") as file:

                    st.download_button(
                        label=f"⬇️ Download {label}",
                        data=file.read(),
                        file_name=file_path.name,
                        mime=(
                            "application/json"
                            if report_type == "json_report"
                            else "text/csv"
                        ),
                        use_container_width=True
                    )

                st.caption(str(file_path))

            else:

                st.warning(f"{label} file was not found.")

        st.divider()

        st.subheader("Raw JSON Response")

        st.json(last_result)

    else:

        st.info("Run an evaluation to generate and download reports.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "LLM Evaluation Agent | Built with Python, FastAPI, "
    "Streamlit, and Ollama"
)
