import streamlit as st

import pandas as pd

import numpy as np

import plotly.graph_objects as go

import joblib

from sklearn.utils.validation import check_is_fitted

from sklearn.exceptions import NotFittedError

from pathlib import Path

import textwrap

# ============================================================

# PAGE CONFIGURATION

# ============================================================

st.set_page_config(

    page_title="EcoFlow | Wastewater Intelligence",

    page_icon="💧",

    layout="wide",

    initial_sidebar_state="expanded",

)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "wastewater_model_Cleaned data.csv"

FEATURE_SCALER_FILE = BASE_DIR / "feature_scaler.pkl"

TARGET_SCALER_FILE = BASE_DIR / "target_scaler.pkl"

LSTM_FILE = BASE_DIR / "lstm_wastewater_model.keras"

TCN_FILE = BASE_DIR / "tcn_wastewater_model.keras"

# Isolation Forest Model Assets

ISO_MODEL_FILE = BASE_DIR / "isolation_forest_model.pkl"

ISO_SCALER_FILE = BASE_DIR / "iso_scaler.pkl"
ISO_CLIP_FILE = BASE_DIR / "iso_clip_bounds.pkl"

TARGET_COLUMNS = ["T1_O2", "T1_NH4", "T1_PO4"]  # Outputs supported by the saved forecasting models

DISPLAY_PARAMETERS = ["TEMPERATURE", "T1_PO4", "T1_NH4", "T1_O2"]  # Parameters shown in dashboard filters

PARAMETER_LABELS = {"TEMPERATURE": "Temperature", "T1_PO4": "PO4", "T1_NH4": "NH4", "T1_O2": "O2"}

FEATURE_COLUMNS = [

    "T1_O2", "T1_NH4", "T1_PO4", "TEMPERATURE",

    "IN_Q", "IN_METAL_Q", "METAL_Q", "MAX_CF",

]

WINDOW_SIZE = 60

# ============================================================

# STYLING — inspired by the supplied dashboard reference

# ============================================================

st.markdown(
    textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --theme-bg: #211A2C;
        --theme-panel: #30243B;
        --theme-panel-alt: #3B2A45;
        --theme-border: #65485D;
        --theme-sakura: #F6A6C1;
        --theme-sunset: #FFB86B;
        --theme-lilac: #C5A3E8;
        --theme-coral: #F28C7B;
        --theme-mint: #A8D5BA;
        --theme-text: #FFF0E8;
        --theme-muted: #D8C0D0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main {
        background: var(--theme-bg);
        color: var(--theme-text);
    }

    [data-testid="stHeader"] {
        background: rgba(26, 27, 38, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #191321;
        border-right: 1px solid var(--theme-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--theme-text);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: var(--theme-sakura); font-size: .72rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .65rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .42rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: #271D32; border: 1px solid #4B354D; border-radius: 11px;
        padding: .62rem .75rem; margin: 0;
        transition: border-color .18s ease, background .18s ease, transform .18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #30243B; border-color: var(--theme-sunset); transform: translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(100deg, rgba(246,166,193,.20), rgba(255,184,107,.14));
        border-color: var(--theme-sakura);
        box-shadow: inset 3px 0 0 var(--theme-sakura), 0 0 13px rgba(246,166,193,.10);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p { font-weight: 600; font-size: .88rem; }

    .block-container {
        width: 100%;
        max-width: 100%;
        padding: 1.25rem clamp(1rem, 2vw, 2rem) 2rem;
    }

    .top-header {
        width: 100%;
        box-sizing: border-box;
        background: linear-gradient(115deg, #30243B 0%, #493047 100%);
        border: 1px solid var(--theme-border);
        border-left: 4px solid var(--theme-sakura);
        border-radius: 16px;
        padding: 22px 26px;
        margin: 0 0 18px 0;
        overflow-wrap: anywhere;
        box-shadow: 0 0 22px rgba(246, 166, 193, 0.10);
    }

    .eyebrow {
        color: var(--theme-sakura);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
    }

    .main-title {
        font-size: clamp(1.35rem, 2.2vw, 2rem);
        line-height: 1.3;
        font-weight: 700;
        color: #FFF5EF;
        margin: 4px 0 0 0;
    }

    .sub-title {
        color: var(--theme-muted);
        font-size: .94rem;
        margin-top: 7px;
    }

    .metric-card {
        width: 100%;
        height: 100%;
        min-height: 132px;
        box-sizing: border-box;
        background: var(--theme-panel);
        border: 1px solid var(--theme-border);
        border-top: 4px solid var(--theme-sakura);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, .18);
        overflow-wrap: anywhere;
    }

    .metric-label {
        color: var(--theme-muted);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .035em;
        margin-bottom: 10px;
    }

    .metric-value {
        color: var(--theme-sakura);
        font-size: clamp(1.3rem, 1.8vw, 1.8rem);
        line-height: 1.2;
        font-weight: 700;
    }

    .metric-note {
        color: var(--theme-muted);
        font-size: .76rem;
        line-height: 1.45;
        margin-top: 9px;
    }

    .section-card {
        width: 100%;
        box-sizing: border-box;
        background: var(--theme-panel);
        border: 1px solid var(--theme-border);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        overflow: hidden;
    }

    .section-heading {
        color: #FFF5EF;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .section-caption {
        color: var(--theme-muted);
        font-size: .78rem;
        margin-bottom: 10px;
    }

    div[data-testid="stMetric"] {
        background: var(--theme-panel);
        border: 1px solid var(--theme-border);
        border-top: 3px solid var(--theme-lilac);
        padding: 14px 16px;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--theme-text);
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: var(--theme-panel-alt);
        border-color: var(--theme-border);
        color: var(--theme-text);
    }

    input, textarea {
        color: var(--theme-text) !important;
    }

    .stButton button, .stDownloadButton button {
        border-radius: 9px;
        border: 1px solid var(--theme-sunset);
        background: #3B2A45;
        color: #FFF5EF;
        transition: all .2s ease;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        border-color: var(--theme-sakura);
        color: var(--theme-sakura);
        box-shadow: 0 0 12px rgba(255, 184, 107, .18);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--theme-border);
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: var(--theme-border);
    }

    p, label, li, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--theme-text);
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: .8rem;
            padding-right: .8rem;
        }
        .top-header {
            padding: 17px;
        }
        .metric-card {
            min-height: 115px;
            padding: 14px;
        }
    }
    </style>
    """),
    unsafe_allow_html=True,
)

# ============================================================

# DATA / MODEL LOADING

# ============================================================

@st.cache_data(show_spinner=False)

def load_dataset(path_string):

    data = pd.read_csv(path_string)

    if "date" not in data.columns:

        raise ValueError("The dataset must contain a 'date' column.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)

    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in data.columns:

        if col != "date":

            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data

@st.cache_resource(show_spinner="Loading trained models and scalers...")

def load_models_and_scalers(feature_path, target_path, lstm_path, tcn_path):

    # Import TensorFlow only when a model page is opened, so the overview can load independently.

    from tensorflow.keras.models import load_model

    feature_scaler = joblib.load(feature_path)

    target_scaler = joblib.load(target_path)

    lstm_model = load_model(lstm_path, compile=False)

    tcn_model = load_model(tcn_path, compile=False)

    return feature_scaler, target_scaler, lstm_model, tcn_model

@st.cache_resource(show_spinner="Loading Isolation Forest Anomaly Detection Model...")
def load_isolation_forest_assets(iso_model_path, iso_scaler_path, iso_clip_path):
    iso_model = joblib.load(iso_model_path)
    iso_scaler = joblib.load(iso_scaler_path)
    iso_clip_bounds = joblib.load(iso_clip_path)
    return iso_model, iso_scaler, iso_clip_bounds

def metric_card(label, value, note=""):

    """Render a responsive metric card with HTML parsed as markup."""

    card_html = f"""<div class="metric-card">

<div class="metric-label">{label}</div>

<div class="metric-value">{value}</div>

<div class="metric-note">{note}</div>

</div>"""

    st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)

def style_chart(fig, height=300):

    fig.update_layout(

        template="plotly_white",

        height=height,

        margin=dict(l=12, r=12, t=42, b=12),

        paper_bgcolor="rgba(26,27,38,0)",

        plot_bgcolor="#30243B",

        font=dict(family="Inter, sans-serif", color="#D8C0D0", size=11),

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),

        hovermode="x unified",

    )

    fig.update_xaxes(showgrid=False, linecolor="#65485D")

    fig.update_yaxes(gridcolor="#493747", zeroline=False)

    return fig

def create_sequences(features, targets, window_size=WINDOW_SIZE):

    X_seq, y_seq = [], []

    for i in range(window_size, len(features)):

        X_seq.append(features[i-window_size:i])

        y_seq.append(targets[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)

def build_test_predictions(data, feature_scaler, target_scaler, model, limit=1200):

    """Predict a recent chronological test slice using the saved model and scalers."""

    split_idx = int(len(data) * 0.8)

    if split_idx <= WINDOW_SIZE or len(data) - split_idx <= 1:

        raise ValueError("Not enough records to create the chronological test sequences.")

    feature_values = data[FEATURE_COLUMNS].astype(float).to_numpy()

    target_values = data[TARGET_COLUMNS].astype(float).to_numpy()

    X_train = feature_values[:split_idx]

    X_test = feature_values[split_idx:]

    y_train = target_values[:split_idx]

    y_test = target_values[split_idx:]

    X_train_scaled = feature_scaler.transform(X_train)

    X_test_scaled = feature_scaler.transform(X_test)

    y_train_scaled = target_scaler.transform(y_train)

    y_test_scaled = target_scaler.transform(y_test)

    X_test_with_history = np.concatenate(

        [X_train_scaled[-WINDOW_SIZE:], X_test_scaled], axis=0

    )

    y_test_with_history = np.concatenate(

        [y_train_scaled[-WINDOW_SIZE:], y_test_scaled], axis=0

    )

    X_test_seq, y_test_seq = create_sequences(

        X_test_with_history, y_test_with_history, WINDOW_SIZE

    )

    # Limit plotting/evaluation workload while retaining the most recent test observations.

    if limit and len(X_test_seq) > limit:

        X_test_seq = X_test_seq[-limit:]

        y_test_seq = y_test_seq[-limit:]

        dates = data["date"].iloc[-limit:].reset_index(drop=True)

    else:

        dates = data["date"].iloc[split_idx:].reset_index(drop=True)

    pred_scaled = model.predict(X_test_seq, verbose=0)

    pred = target_scaler.inverse_transform(np.asarray(pred_scaled))

    actual = target_scaler.inverse_transform(np.asarray(y_test_seq))

    return dates.iloc[-len(actual):].reset_index(drop=True), actual, pred


def forecast_future_horizon(data, feature_scaler, target_scaler, model, horizon_minutes):
    """Recursive one-step forecasting; hold non-target inputs at latest observed values."""
    recent = data.dropna(subset=FEATURE_COLUMNS).tail(WINDOW_SIZE).copy()
    if len(recent) < WINDOW_SIZE:
        raise ValueError(f"At least {WINDOW_SIZE} complete observations are required.")
    gaps = recent["date"].diff().dropna().dt.total_seconds().div(60)
    interval_minutes = float(gaps[gaps > 0].median()) if not gaps.empty and (gaps > 0).any() else 2.0
    if not np.isfinite(interval_minutes) or interval_minutes <= 0:
        interval_minutes = 2.0
    steps = max(1, int(round(float(horizon_minutes) / interval_minutes)))
    effective_minutes = steps * interval_minutes
    current_features = recent[FEATURE_COLUMNS].astype(float).to_numpy()
    sequence = feature_scaler.transform(current_features)
    last_date = recent["date"].iloc[-1]
    forecast_rows = []
    target_indices = {name: FEATURE_COLUMNS.index(name) for name in TARGET_COLUMNS}
    for step in range(1, steps + 1):
        predicted_scaled = np.asarray(model.predict(sequence[np.newaxis, :, :], verbose=0))
        predicted = target_scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0]
        next_raw = current_features[-1].copy()
        for j, target_name in enumerate(TARGET_COLUMNS):
            next_raw[target_indices[target_name]] = predicted[j]
        forecast_rows.append({
            "date": last_date + pd.Timedelta(minutes=interval_minutes * step),
            **{name: float(predicted[j]) for j, name in enumerate(TARGET_COLUMNS)},
        })
        next_scaled = feature_scaler.transform(next_raw.reshape(1, -1))
        sequence = np.vstack([sequence[1:], next_scaled])
        current_features = np.vstack([current_features[1:], next_raw])
    result = pd.DataFrame(forecast_rows)
    result.attrs["effective_horizon_minutes"] = float(effective_minutes)
    result.attrs["sampling_interval_minutes"] = float(interval_minutes)
    return result

def make_line_chart(data, parameter, title, limit=1500, threshold_lines=False):

    view = data.tail(limit)

    fig = go.Figure()

    fig.add_trace(go.Scatter(

        x=view["date"], y=view[parameter], mode="lines",

        name="Observed", line=dict(color="#F6A6C1", width=1.8),

    ))

    fig.update_layout(title=title, xaxis_title="Timestamp", yaxis_title=parameter)

    if threshold_lines:

        # Statistical reference lines only; not regulatory compliance limits.

        q1, q3 = data[parameter].quantile([0.25, 0.75])

        iqr = q3 - q1

        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        fig.add_hline(y=upper, line_dash="dash", line_color="#F28C7B",

                      annotation_text="Statistical upper boundary")

        fig.add_hline(y=lower, line_dash="dash", line_color="#F6A6C1",

                      annotation_text="Statistical lower boundary")

    return style_chart(fig, 320)

# ============================================================

# SAFE STARTUP / VALIDATION

# ============================================================

st.sidebar.markdown("## 💧 EcoFlow Intelligence")

st.sidebar.caption("Industrial wastewater monitoring")

st.sidebar.markdown("---")

if not DATA_FILE.exists():

    st.error(f"Dataset not found: {DATA_FILE.name}")

    st.info("Place the cleaned CSV in the same folder as Apps.py.")

    st.stop()

try:

    df = load_dataset(str(DATA_FILE))

except Exception as exc:

    st.error(f"Could not load the dataset: {exc}")

    st.stop()

missing = [c for c in ["date"] + FEATURE_COLUMNS if c not in df.columns]

if missing:

    st.error("The dataset is missing required columns: " + ", ".join(missing))

    st.stop()

# Exclude rows with missing model feature/target values from model-based views.

model_df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

if model_df.empty:

    st.error("No complete records are available for the model features.")

    st.stop()

page = st.sidebar.radio(

    "NAVIGATION",

    [

        "Executive Overview",

        "Historical Monitoring",

        "LSTM vs TCN Forecasting",

        "Anomaly Detection",

        "Model Evaluation",

        "Data Explorer",

    ],

)

st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset Information")

st.sidebar.caption(f"Records: {len(df):,}")

st.sidebar.caption(f"Sampling interval: approximately 2 minutes")

st.sidebar.caption(f"First record: {df['date'].min().strftime('%d %b %Y')}")

st.sidebar.caption(f"Last record: {df['date'].max().strftime('%d %b %Y')}")

st.sidebar.caption("Forecasts are model-based, not live sensor telemetry.")

# Header

st.markdown(

    textwrap.dedent("""

    <div class="top-header">

      <div class="eyebrow">EcoFlow Intelligence Platform</div>

      <div class="main-title">Industrial Wastewater Quality Monitoring</div>

      <div class="sub-title">Machine-learning-based trend prediction and operational insight</div>

    </div>

    """),

    unsafe_allow_html=True,

)

# ============================================================

# PAGE 1 — EXECUTIVE OVERVIEW

# ============================================================

if page == "Executive Overview":

    st.markdown("### Executive Overview")

    st.caption("A high-level summary of historical wastewater process data.")

    latest = model_df.iloc[-1]

    avg_temp = model_df["TEMPERATURE"].mean()

    avg_inflow = model_df["IN_Q"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        metric_card("TOTAL RECORDS", f"{len(df):,}", "Rows in cleaned dataset")

    with col2:

        metric_card("LATEST T1_O2", f"{latest['T1_O2']:.3f}", "Latest recorded value")

    with col3:

        metric_card("LATEST T1_NH4", f"{latest['T1_NH4']:.3f}", "Latest recorded value")

    with col4:

        metric_card("AVG TEMPERATURE", f"{avg_temp:.2f} °C", "Across available records")

    with col5:

        metric_card("AVG INFLOW RATE", f"{avg_inflow:,.2f}", "IN_Q dataset units")

    st.markdown("")

    left, right = st.columns([1.15, 1])

    with left:

        st.markdown('<div class="section-card"><div class="section-heading">Historical Water Quality Trend</div><div class="section-caption">Select a parameter to explore recent readings.</div>', unsafe_allow_html=True)

        parameter = st.selectbox("Water quality parameter", [c for c in DISPLAY_PARAMETERS if c in model_df.columns], format_func=lambda c: PARAMETER_LABELS[c], key="overview_parameter")

        days = st.selectbox("Display period", ["Last 24 hours", "Last 7 days", "Last 30 days", "All records"], index=2)

        if days == "Last 24 hours":

            start = model_df["date"].max() - pd.Timedelta(hours=24)

            chart_df = model_df[model_df["date"] >= start]

        elif days == "Last 7 days":

            start = model_df["date"].max() - pd.Timedelta(days=7)

            chart_df = model_df[model_df["date"] >= start]

        elif days == "Last 30 days":

            start = model_df["date"].max() - pd.Timedelta(days=30)

            chart_df = model_df[model_df["date"] >= start]

        else:

            chart_df = model_df

        if chart_df.empty:

            chart_df = model_df.tail(500)

        st.plotly_chart(make_line_chart(chart_df, parameter, f"{parameter} — observed history", limit=3000), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown('<div class="section-card"><div class="section-heading">Latest Process Readings</div><div class="section-caption">Most recent valid record in the dataset.</div>', unsafe_allow_html=True)

        summary = pd.DataFrame({

            "Parameter": ["TEMPERATURE", "T1_PO4", "T1_NH4", "T1_O2"],

            "Latest value": [

                latest["TEMPERATURE"], latest["T1_PO4"], latest["T1_NH4"], latest["T1_O2"]

            ],

        })

        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown(f"**Last timestamp:** {latest['date'].strftime('%d %b %Y, %H:%M UTC')}")

        st.info("This dashboard currently displays historical dataset readings. It is not connected to a live plant sensor feed.")

        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns([1, 1])

    with lower_left:

        st.markdown('<div class="section-card"><div class="section-heading">Parameter Distribution</div><div class="section-caption">Distribution across the cleaned dataset.</div>', unsafe_allow_html=True)

        dist_param = st.selectbox("Distribution parameter", [c for c in DISPLAY_PARAMETERS if c in model_df.columns], format_func=lambda c: PARAMETER_LABELS[c], key="dist_parameter")

        fig = go.Figure(go.Histogram(x=model_df[dist_param], nbinsx=40,

                                     marker_color="#FFB86B", opacity=0.88))

        fig.update_layout(title=dist_param, xaxis_title="Observed value", yaxis_title="Frequency")

        st.plotly_chart(style_chart(fig, 270), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with lower_right:

        st.markdown('<div class="section-card"><div class="section-heading">Data Coverage</div><div class="section-caption">Dataset time span and completeness.</div>', unsafe_allow_html=True)

        coverage = pd.DataFrame({

            "Item": ["First timestamp", "Last timestamp", "Available records", "Missing values in model features"],

            "Value": [

                df["date"].min().strftime("%d %b %Y %H:%M UTC"),

                df["date"].max().strftime("%d %b %Y %H:%M UTC"),

                f"{len(df):,}",

                f"{int(df[FEATURE_COLUMNS].isna().sum().sum()):,}",

            ],

        })

        st.dataframe(coverage, hide_index=True, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================

# PAGE 2 — HISTORICAL MONITORING

# ============================================================

elif page == "Historical Monitoring":

    st.markdown("### Historical Monitoring")

    st.caption("Inspect recorded process variables over a chosen period.")

    p1, p2 = st.columns([1, 1])

    with p1:

        parameter = st.selectbox("Select parameter", [c for c in DISPLAY_PARAMETERS if c in df.columns], format_func=lambda c: PARAMETER_LABELS[c], key="historical_parameter")

    with p2:

        point_count = st.select_slider("Number of recent observations", options=[250, 500, 1000, 2000, 3000, 5000], value=1000)

    view = df.dropna(subset=[parameter]).tail(point_count)

    st.plotly_chart(make_line_chart(view, parameter, f"Historical trend: {parameter}", limit=point_count, threshold_lines=True), use_container_width=True)

    st.caption("Dashed lines are statistical IQR outlier boundaries, not regulatory or environmental compliance limits.")

    st.markdown("#### Recent readings")

    st.dataframe(view[["date", parameter]].tail(100).sort_values("date", ascending=False), hide_index=True, use_container_width=True)

# ============================================================

# MODEL LOAD — only required for model-related pages

# ============================================================

elif page in ["LSTM vs TCN Forecasting", "Model Evaluation"]:

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    missing_assets = [p.name for p in [FEATURE_SCALER_FILE, TARGET_SCALER_FILE, LSTM_FILE, TCN_FILE] if not p.exists()]

    if missing_assets:

        st.error("Model files not found: " + ", ".join(missing_assets))

        st.info("Place both .keras models and both .pkl scalers in the same folder as Apps.py.")

        st.stop()

    try:

        feature_scaler, target_scaler, lstm_model, tcn_model = load_models_and_scalers(

            str(FEATURE_SCALER_FILE), str(TARGET_SCALER_FILE), str(LSTM_FILE), str(TCN_FILE)

        )

    except Exception as exc:

        st.error("The models or scalers could not be loaded.")

        st.code(str(exc))

        st.warning("If the error mentions NumPy, TensorFlow, or a compiled module, fix the Python environment first. Do not retrain or overwrite the saved models yet.")

        st.stop()

    if page == "LSTM vs TCN Forecasting":

        st.markdown("### LSTM vs TCN Forecasting")

        st.caption("Predictions are generated by the saved trained models on a recent chronological test segment.")

        target_choice = st.selectbox("Target parameter", TARGET_COLUMNS)

        max_points = st.select_slider("Recent test observations to predict", options=[100, 250, 500, 800, 1200], value=250)

        model_choice = st.radio("Model to display", ["Compare LSTM and TCN", "LSTM only", "TCN only"], horizontal=True)

        try:

            with st.spinner("Generating predictions from the trained models..."):

                dates_lstm, actual_lstm, pred_lstm = build_test_predictions(

                    model_df, feature_scaler, target_scaler, lstm_model, limit=max_points

                )

                dates_tcn, actual_tcn, pred_tcn = build_test_predictions(

                    model_df, feature_scaler, target_scaler, tcn_model, limit=max_points

                )

            target_idx = TARGET_COLUMNS.index(target_choice)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=dates_lstm, y=actual_lstm[:, target_idx],

                                     name="Actual", mode="lines",

                                     line=dict(color="#F6A6C1", width=2)))

            if model_choice in ["Compare LSTM and TCN", "LSTM only"]:

                fig.add_trace(go.Scatter(x=dates_lstm, y=pred_lstm[:, target_idx],

                                         name="LSTM prediction", mode="lines",

                                         line=dict(color="#FFB86B", width=1.7, dash="dash")))

            if model_choice in ["Compare LSTM and TCN", "TCN only"]:

                fig.add_trace(go.Scatter(x=dates_tcn, y=pred_tcn[:, target_idx],

                                         name="TCN prediction", mode="lines",

                                         line=dict(color="#FFB86B", width=1.7, dash="dot")))

            fig.update_layout(title=f"Actual vs predicted {target_choice}",

                              xaxis_title="Timestamp", yaxis_title=target_choice)

            st.plotly_chart(style_chart(fig, 430), use_container_width=True)

            m1, m2, m3 = st.columns(3)

            with m1:

                metric_card("OBSERVED MEAN", f"{np.mean(actual_lstm[:, target_idx]):.4f}", target_choice)

            with m2:

                metric_card("LSTM MAE", f"{mean_absolute_error(actual_lstm[:, target_idx], pred_lstm[:, target_idx]):.4f}", "Lower is better")

            with m3:

                metric_card("TCN MAE", f"{mean_absolute_error(actual_tcn[:, target_idx], pred_tcn[:, target_idx]):.4f}", "Lower is better")

            st.caption("The comparison above evaluates one-step predictions on a held-out historical test segment.")

            st.markdown("---")
            st.markdown("#### Future forecast horizon")
            st.caption("Choose how far ahead to project from the latest available reading.")
            horizon_minutes = st.select_slider(
                "Forecast ahead", options=[30, 45, 60], value=30,
                format_func=lambda value: f"{value} minutes", key="forecast_horizon_minutes",
            )
            future_model_choice = st.radio(
                "Future forecast model", ["Compare LSTM and TCN", "LSTM", "TCN"],
                horizontal=True, key="future_forecast_model_choice",
            )
            try:
                forecast_map = {}
                models_to_run = []
                if future_model_choice in ["Compare LSTM and TCN", "LSTM"]:
                    models_to_run.append(("LSTM", lstm_model))
                if future_model_choice in ["Compare LSTM and TCN", "TCN"]:
                    models_to_run.append(("TCN", tcn_model))
                for forecast_name, forecast_model in models_to_run:
                    forecast_map[forecast_name] = forecast_future_horizon(
                        model_df, feature_scaler, target_scaler, forecast_model, horizon_minutes
                    )
                effective_minutes = next(iter(forecast_map.values())).attrs["effective_horizon_minutes"]
                history_view = model_df.dropna(subset=FEATURE_COLUMNS).tail(60)
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=history_view["date"], y=history_view[target_choice], name="Recent observed",
                    mode="lines", line=dict(color="#FFF0E8", width=2),
                ))
                for forecast_name, forecast_data in forecast_map.items():
                    fig_future.add_trace(go.Scatter(
                        x=forecast_data["date"], y=forecast_data[target_choice],
                        name=f"{forecast_name} forecast", mode="lines+markers",
                        line=dict(color="#F6A6C1" if forecast_name == "LSTM" else "#C5A3E8", width=2, dash="dash"),
                        marker=dict(size=4),
                    ))
                fig_future.update_layout(
                    title=f"{target_choice} forecast — approximately {effective_minutes:g} minutes ahead",
                    xaxis_title="Timestamp", yaxis_title=target_choice,
                )
                st.plotly_chart(style_chart(fig_future, 400), use_container_width=True)
                summary = []
                for forecast_name, forecast_data in forecast_map.items():
                    final_row = forecast_data.iloc[-1]
                    summary.append({
                        "Model": forecast_name,
                        "Target": target_choice,
                        "Forecast timestamp (UTC)": final_row["date"].strftime("%d %b %Y %H:%M"),
                        "Predicted value": round(float(final_row[target_choice]), 4),
                        "Requested horizon (min)": horizon_minutes,
                        "Effective horizon (min)": round(effective_minutes, 1),
                    })
                st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
                st.info(
                    "Forecast assumption: predicted O₂/NH₄/PO₄ values are fed back recursively into the next step. "
                    "Future temperature and flow-related inputs are held at their latest observed values because future "
                    "inputs are not available. Treat this as an exploratory projection, not a guaranteed plant outcome."
                )
            except Exception as forecast_exc:
                st.error("Future forecasting could not be generated with the current model and scaler assets.")
                st.code(str(forecast_exc))

        except Exception as exc:

            st.error("Prediction could not be generated. Check that the feature order and scalers match the training notebooks.")

            st.code(str(exc))

    elif page == "Model Evaluation":

        st.markdown("### Model Performance & Evaluation")

        st.caption("Metrics are calculated from predictions against held-out chronological test observations.")

        try:

            with st.spinner("Evaluating LSTM and TCN on recent test observations..."):

                _, actual_lstm, pred_lstm = build_test_predictions(model_df, feature_scaler, target_scaler, lstm_model, limit=5000)

                _, actual_tcn, pred_tcn = build_test_predictions(model_df, feature_scaler, target_scaler, tcn_model, limit=5000)

            rows = []

            for model_name, actual, predicted in [

                ("LSTM", actual_lstm, pred_lstm), ("TCN", actual_tcn, pred_tcn)

            ]:

                for i, target in enumerate(TARGET_COLUMNS):

                    rows.append({

                        "Model": model_name,

                        "Target": target,

                        "MAE": mean_absolute_error(actual[:, i], predicted[:, i]),

                        "RMSE": np.sqrt(mean_squared_error(actual[:, i], predicted[:, i])),

                        "R²": r2_score(actual[:, i], predicted[:, i]),

                    })

            metrics = pd.DataFrame(rows)

            st.dataframe(metrics.round(4), hide_index=True, use_container_width=True)

            # MAE comparison chart using the EcoFlow pastel colour palette.

            fig = go.Figure()

            mae_model_series = [

                ("LSTM", actual_lstm, pred_lstm, "#FFB86B"),  # muted blue

                ("TCN", actual_tcn, pred_tcn, "#C5A3E8"),    # teal

            ]

            for model_name, actual, predicted, bar_color in mae_model_series:

                mae_values = [

                    mean_absolute_error(actual[:, i], predicted[:, i])

                    for i in range(len(TARGET_COLUMNS))

                ]

                fig.add_trace(

                    go.Bar(

                        name=model_name,

                        x=TARGET_COLUMNS,

                        y=mae_values,

                        marker=dict(

                            color=bar_color,

                            line=dict(color="#65485D", width=1),

                        ),

                        hovertemplate=(

                            "<b>%{fullData.name}</b><br>"

                            "Target: %{x}<br>"

                            "MAE: %{y:.4f}<extra></extra>"

                        ),

                    )

                )

            fig.update_layout(

                title=dict(

                    text="MAE Comparison by Target (Lower Is Better)",

                    font=dict(color="#FFF5EF", size=16),

                    x=0,

                    xanchor="left",

                ),

                template="plotly_white",

                barmode="group",

                bargap=0.28,

                paper_bgcolor="#30243B",

                plot_bgcolor="#211A2C",

                font=dict(family="Inter, sans-serif", color="#FFF0E8", size=12),

                legend=dict(

                    orientation="h",

                    yanchor="bottom",

                    y=1.02,

                    xanchor="left",

                    x=0,

                    bgcolor="rgba(255,255,255,0)",

                ),

                xaxis=dict(

                    title="Target Parameter",

                    showgrid=False,

                    linecolor="#65485D",

                    zeroline=False,

                ),

                yaxis=dict(

                    title="Mean Absolute Error (MAE)",

                    showgrid=True,

                    gridcolor="#493747",

                    zeroline=False,

                    rangemode="tozero",

                ),

                margin=dict(l=45, r=20, t=70, b=45),

                height=360,

            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.caption(

                "Metrics use up to the latest 5,000 test observations to keep the dashboard responsive; "

                "these may differ from full-test metrics reported in your notebooks."

            )

        except Exception as exc:

            st.error("Evaluation could not be completed.")

            st.code(str(exc))

# ============================================================

# PAGE 4 — ANOMALY DETECTION (ISOLATION FOREST INTEGRATED)

# ============================================================

elif page == "Anomaly Detection":

    st.markdown("### Anomaly Detection")

    st.caption("Machine learning outlier detection using trained Isolation Forest alongside baseline statistical screening.")

    # Load Isolation Forest Model and Scaler

    iso_missing = [p.name for p in [ISO_MODEL_FILE, ISO_SCALER_FILE, ISO_CLIP_FILE] if not p.exists()]

    if iso_missing:

        st.warning(f"Isolation Forest model files not found: {', '.join(iso_missing)}. Displaying statistical screening baseline.")

        iso_model_loaded = False

    else:

        try:

            iso_forest, iso_scaler, iso_clip_bounds = load_isolation_forest_assets(
                str(ISO_MODEL_FILE), str(ISO_SCALER_FILE), str(ISO_CLIP_FILE)
            )

            iso_model_loaded = True

        except Exception as exc:

            st.error(f"Could not load Isolation Forest model: {exc}")

            iso_model_loaded = False

    detection_mode = st.radio(

        "Detection Strategy", 

        ["Isolation Forest ML Model", "Statistical IQR Baseline"], 

        horizontal=True,

        index=0 if iso_model_loaded else 1

    )

    p1, p2 = st.columns([1, 1])

    with p1:

        parameter = st.selectbox("Parameter to display", [c for c in DISPLAY_PARAMETERS if c in df.columns], format_func=lambda c: PARAMETER_LABELS[c], key="anomaly_parameter")

    with p2:

        point_count = st.select_slider("Recent observations to inspect", options=[500, 1000, 2000, 5000, 10000], value=2000)

    data = model_df.tail(point_count).copy()

    if data.empty:

        st.warning("No valid records are available for anomaly detection.")

        st.stop()

    if detection_mode == "Isolation Forest ML Model" and iso_model_loaded:

        # Match the exact feature order and clipping used during Isolation Forest training.
        feature_cols = FEATURE_COLUMNS.copy()

        try:
            missing_iso_features = [column for column in feature_cols if column not in data.columns]
            if missing_iso_features:
                raise ValueError(
                    "Dataset is missing Isolation Forest features: "
                    + ", ".join(missing_iso_features)
                )

            X_iso = data[feature_cols].copy()
            for column in feature_cols:
                bounds = iso_clip_bounds[column]
                X_iso[column] = X_iso[column].clip(
                    lower=bounds["lower"],
                    upper=bounds["upper"],
                )

            X_scaled = iso_scaler.transform(X_iso)

            try:

                check_is_fitted(iso_forest)

            except NotFittedError:

                st.error(

                    "The saved Isolation Forest model is not fitted. "

                    "Load the fitted model created during training; this dashboard "

                    "will not retrain it automatically."

                )

                st.stop()

            data["anomaly_pred"] = iso_forest.predict(X_scaled)

            data["Anomaly flag"] = data["anomaly_pred"] == -1

            data["anomaly_score"] = iso_forest.decision_function(X_scaled)

        except Exception as exc:

            st.error(f"Isolation Forest prediction failed: {exc}")

            st.stop()

        total_anomalies = int(data["Anomaly flag"].sum())

        anomaly_pct = (total_anomalies / len(data)) * 100

        # These are illustrative screening thresholds, not verified regulatory limits.

        data["critical_violation"] = (data["T1_O2"] < 0.5) | (data["T1_NH4"] > 5.0)

        critical_violations = int(data["critical_violation"].sum())

        a, b, c = st.columns(3)

        with a:

            metric_card(

                "DETECTED ML ANOMALIES",

                f"{total_anomalies:,}",

                f"{anomaly_pct:.2f}% of inspected records",

            )

        with b:

            metric_card(

                "AVG ANOMALY SCORE",

                f"{data['anomaly_score'].mean():.4f}",

                "Lower scores generally indicate more anomalous observations",

            )

        with c:

            metric_card(

                "SCREENING THRESHOLD FLAGS",

                f"{critical_violations:,}",

                "Illustrative thresholds only; not verified compliance limits",

            )

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=data["date"],

                y=data[parameter],

                name="Observed",

                mode="lines",

                line=dict(color="#F6A6C1", width=1.3),

            )

        )

        anomalies = data[data["Anomaly flag"]]

        fig.add_trace(

            go.Scatter(

                x=anomalies["date"],

                y=anomalies[parameter],

                name="Isolation Forest anomaly",

                mode="markers",

                marker=dict(color="#F28C7B", size=7, symbol="x"),

            )

        )

        fig.update_layout(

            title=f"Isolation Forest Anomaly Detection: {parameter}",

            xaxis_title="Timestamp",

            yaxis_title=parameter,

        )

        st.plotly_chart(style_chart(fig, 400), use_container_width=True)

        st.markdown(

            '<div class="section-card"><div class="section-heading">'

            'Anomaly Decision Score Distribution</div>'

            '<div class="section-caption">Decision scores assigned by the saved Isolation Forest model.</div>',

            unsafe_allow_html=True,

        )

        fig_score = go.Figure()

        fig_score.add_trace(

            go.Histogram(

                x=data["anomaly_score"],

                nbinsx=40,

                marker_color="#C5A3E8",

                opacity=0.85,

            )

        )

        fig_score.update_layout(

            title="Decision Score Distribution",

            xaxis_title="Decision Score",

            yaxis_title="Frequency",

        )

        st.plotly_chart(style_chart(fig_score, 250), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Latest ML-flagged anomalies")

        display_cols = list(dict.fromkeys(["date", parameter, "anomaly_score", "T1_O2", "T1_NH4"]))

        display_cols = [column for column in display_cols if column in data.columns]

        flagged = data.loc[data["Anomaly flag"], display_cols].copy()

        st.dataframe(
            flagged.tail(100).sort_values("date", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### Anomaly notifications & recommended actions")
        st.caption(
            "These are investigation prompts based on unusual patterns, not confirmed faults or regulatory violations. "
            "Verify readings and follow your plant's approved procedures."
        )
        notification_rows = data.loc[data["Anomaly flag"]].sort_values("date", ascending=False)
        if notification_rows.empty:
            st.success("No Isolation Forest anomalies were detected in the selected observation window.")
        else:
            st.caption(f"Showing all {len(notification_rows)} detected anomalies in the selected observation window.")
            for _, anomaly_row in notification_rows.iterrows():
                readings = []
                for col, label, fmt in [
                    ("T1_O2", "O₂", ".3f"), ("T1_NH4", "NH₄", ".3f"),
                    ("T1_PO4", "PO₄", ".3f"), ("TEMPERATURE", "Temperature", ".2f")
                ]:
                    if col in anomaly_row.index and pd.notna(anomaly_row[col]):
                        readings.append(f"{label}: {format(float(anomaly_row[col]), fmt)}")
                reasons, actions = [], []
                # Illustrative screening values only; these are not regulatory limits.
                if pd.notna(anomaly_row.get("T1_O2")) and float(anomaly_row["T1_O2"]) < 0.5:
                    reasons.append("O₂ is below the dashboard's illustrative screening value.")
                    actions.extend([
                        "Verify the dissolved-oxygen sensor and calibration using the approved procedure.",
                        "Inspect aeration equipment and airflow indicators; compare nearby readings before process changes.",
                    ])
                if pd.notna(anomaly_row.get("T1_NH4")) and float(anomaly_row["T1_NH4"]) > 5.0:
                    reasons.append("NH₄ is above the dashboard's illustrative screening value.")
                    actions.extend([
                        "Confirm the NH₄ reading with the approved sensor-check or sampling procedure.",
                        "Review aeration/nitrification conditions and recent influent loading with the responsible operator.",
                    ])
                if "T1_PO4" in data and pd.notna(anomaly_row.get("T1_PO4")) and anomaly_row["T1_PO4"] > data["T1_PO4"].quantile(0.95):
                    reasons.append("PO₄ is above the 95th percentile of the inspected data window.")
                    actions.extend([
                        "Check PO₄ sensor status or confirm with a sample if required by site SOP.",
                        "Review chemical dosing and phosphorus-removal process records with the operator.",
                    ])
                if "TEMPERATURE" in data and pd.notna(anomaly_row.get("TEMPERATURE")) and anomaly_row["TEMPERATURE"] > data["TEMPERATURE"].quantile(0.95):
                    reasons.append("Temperature is above the 95th percentile of the inspected data window.")
                    actions.extend([
                        "Verify the temperature sensor and compare with upstream/influent readings.",
                        "Check recent process changes or unusual hot-water inflow according to site procedures.",
                    ])
                if not reasons:
                    reasons.append("The combined feature pattern differs from patterns learned by Isolation Forest; the model cannot identify a cause by itself.")
                if not actions:
                    actions = [
                        "Compare this timestamp with nearby readings and recent process logs.",
                        "Verify sensor quality and confirm the observation before operational action.",
                        "Escalate to the responsible operator if the pattern persists.",
                    ]
                stamp = anomaly_row["date"].strftime("%d %b %Y, %H:%M UTC")
                with st.expander(f"🔔 Anomaly • {stamp} • score {float(anomaly_row['anomaly_score']):.4f}"):
                    st.markdown("**Observed readings**")
                    st.write(" · ".join(readings) if readings else "No display readings available.")
                    st.markdown("**Why it was flagged / checks to investigate**")
                    for reason in dict.fromkeys(reasons):
                        st.markdown(f"- {reason}")
                    st.markdown("**Recommended operational checks**")
                    for action in dict.fromkeys(actions):
                        st.markdown(f"- {action}")

    else:

        # Statistical IQR baseline for a selected parameter.

        valid_values = data[parameter].dropna()

        if valid_values.empty:

            st.warning(f"No valid values are available for {parameter}.")

            st.stop()

        q1, q3 = valid_values.quantile([0.25, 0.75])

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        data["Anomaly flag"] = (data[parameter] < lower) | (data[parameter] > upper)

        a, b, c = st.columns(3)

        with a:

            metric_card(

                "FLAGGED RECORDS",

                f"{int(data['Anomaly flag'].sum()):,}",

                "Statistical IQR screening",

            )

        with b:

            metric_card("LOWER STATISTICAL BOUND", f"{lower:.4f}", "Q1 − 1.5 × IQR")

        with c:

            metric_card("UPPER STATISTICAL BOUND", f"{upper:.4f}", "Q3 + 1.5 × IQR")

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=data["date"],

                y=data[parameter],

                name="Observed",

                mode="lines",

                line=dict(color="#FFB86B", width=1.3),

            )

        )

        anomalies = data[data["Anomaly flag"]]

        fig.add_trace(

            go.Scatter(

                x=anomalies["date"],

                y=anomalies[parameter],

                name="Statistical outlier",

                mode="markers",

                marker=dict(color="#F28C7B", size=6),

            )

        )

        fig.add_hline(y=upper, line_dash="dash", line_color="#F28C7B")

        fig.add_hline(y=lower, line_dash="dash", line_color="#F6A6C1")

        fig.update_layout(

            title=f"Statistical outlier screening: {parameter}",

            xaxis_title="Timestamp",

            yaxis_title=parameter,

        )

        st.plotly_chart(style_chart(fig, 400), use_container_width=True)

        st.markdown("#### Latest flagged records")

        st.dataframe(

            data.loc[data["Anomaly flag"], ["date", parameter]]

            .tail(100)

            .sort_values("date", ascending=False),

            hide_index=True,

            use_container_width=True,

        )

    st.warning(

        "Anomalies and statistical outliers are screening results, not confirmed environmental "

        "violations. Verify applicable regulatory limits before interpreting compliance."

    )

# ============================================================

# PAGE 5 — DATA EXPLORER

# ============================================================

elif page == "Data Explorer":

    st.markdown("### Data Explorer")

    st.caption("Filter and inspect the cleaned wastewater dataset.")

    col1, col2 = st.columns(2)

    with col1:

        start_date = st.date_input("Start date", value=df["date"].min().date(),

                                   min_value=df["date"].min().date(), max_value=df["date"].max().date())

    with col2:

        end_date = st.date_input("End date", value=df["date"].max().date(),

                                 min_value=df["date"].min().date(), max_value=df["date"].max().date())

    if start_date > end_date:

        st.error("Start date must be on or before end date.")

    else:

        filtered = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

        st.write(f"**{len(filtered):,} records** in the selected period.")

        visible_columns = ["date"] + [column for column in DISPLAY_PARAMETERS if column in filtered.columns]

        st.dataframe(

            filtered[visible_columns],

            hide_index=True,

            use_container_width=True,

            height=440,

        )

        st.download_button(

            "Download filtered data as CSV",

            data=filtered.to_csv(index=False).encode("utf-8"),

            file_name="wastewater_filtered_data.csv",

            mime="text/csv",

        )

st.markdown("---")

st.caption("EcoFlow Intelligence • Academic FYP prototype • Historical dataset and saved-model inference")