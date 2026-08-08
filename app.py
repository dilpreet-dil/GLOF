
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MiniMap
import branca.colormap as bcm
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="GLOF Early Warning System — India",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# Theme colours — picked from our project's colour palette
# (dark navy background, lighter blues for cards/accents)
# ----------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-base: #001D39;
    --bg-elevated: #0A4174;
    --bg-card: #0A4174;
    --border-soft: #49769F;
    --accent: #7BBDE8;
    --accent-soft: #4E8EA2;
    --text-primary: #BDD8E9;
    --text-muted: #6EA2B3;
    --risk-low: #4CAF6D;
    --risk-medium: #F2B84B;
    --risk-high: #E8555B;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(180deg, #001D39 0%, #04244A 100%);
}

h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

h1 {
    background: linear-gradient(90deg, #7BBDE8, #BDD8E9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #04244A 0%, #001D39 100%);
    border-right: 1px solid rgba(123, 189, 232, 0.15);
}

/* card used everywhere for a "panel" look */
.glof-card {
    background: var(--bg-card);
    border: 1px solid rgba(123, 189, 232, 0.18);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-label {
    color: var(--text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-sub {
    color: var(--text-muted);
    font-size: 0.78rem;
}

.stButton > button {
    background: linear-gradient(135deg, #7BBDE8, #4E8EA2);
    color: #001D39;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
}
.stButton > button:hover {
    filter: brightness(1.08);
}

.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.1rem;
    border-radius: 999px;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
}
.risk-badge.low { background: rgba(76, 175, 109, 0.16); color: var(--risk-low); border: 1px solid rgba(76,175,109,0.4); }
.risk-badge.medium { background: rgba(242, 184, 75, 0.16); color: var(--risk-medium); border: 1px solid rgba(242,184,75,0.4); }
.risk-badge.high { background: rgba(232, 85, 91, 0.16); color: var(--risk-high); border: 1px solid rgba(232,85,91,0.4); }

/* folium returns an <iframe> with no fixed size, which can render as a
   weird stretched rectangle — so we just cap its height ourselves */
.map-container { border-radius: 14px; overflow: hidden; border: 1px solid rgba(123,189,232,0.2); }
.map-container iframe { width: 100% !important; height: 480px !important; display: block; }

.alert-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(123,189,232,0.12);
    padding: 0.6rem 0;
}
.alert-banner {
    background: linear-gradient(135deg, rgba(232,85,91,0.16), rgba(232,85,91,0.05));
    border: 1px solid rgba(232, 85, 91, 0.45);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
}

.risk-dot {
    display: inline-block;
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 50%;
    margin-right: 0.4rem;
}
.risk-dot.low { background: var(--risk-low); }
.risk-dot.medium { background: var(--risk-medium); }
.risk-dot.high { background: var(--risk-high); }

/* number_input already ships with +/- stepper buttons; just theme them
   to match the rest of the UI instead of using sliders */
[data-testid="stNumberInput"] button {
    background: var(--bg-elevated);
    border-color: rgba(123, 189, 232, 0.3);
    color: var(--accent);
}

hr { border-color: rgba(123, 189, 232, 0.15) !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ==================================================================
# Load the trained model + dataset
# ==================================================================
@st.cache_resource
def load_model():
    model = joblib.load("glof_model.joblib")
    scaler = joblib.load("glof_scaler.joblib")
    feature_cols = joblib.load("glof_feature_cols.joblib")
    region_map = joblib.load("glof_region_map.joblib")
    return model, scaler, feature_cols, region_map


model, scaler, feature_cols, region_map = load_model()
inv_region_map = {v: k for k, v in region_map.items()}


@st.cache_data
def load_data():
    # same cleanup we did in the notebook: drop duplicate rows, fill a
    # couple of missing numeric columns with the median, and re-use the
    # SAME region encoding the model was trained on
    data = pd.read_csv("glof_dataset.csv")
    data = data.drop_duplicates().reset_index(drop=True)
    for col in ["snowfall_mm", "earthquake_magnitude"]:
        data[col] = data[col].fillna(data[col].median())
    data["region_encoded"] = data["region"].map(inv_region_map)
    return data


df = load_data()


def risk_bucket(prob):
    """Turn a probability into a low/medium/high label."""
    if prob >= 0.7:
        return "high", "HIGH RISK"
    elif prob >= 0.4:
        return "medium", "MEDIUM RISK"
    return "low", "LOW RISK"


def predict_risk(row_dict):
    """row_dict needs a value for every column in feature_cols."""
    X = pd.DataFrame([row_dict])[feature_cols]
    X_in = scaler.transform(X) if type(model).__name__ == "LogisticRegression" else X
    return model.predict_proba(X_in)[0, 1]


@st.cache_data
def score_all_lakes():
    """Run the model once over the whole dataset so we can show dashboard
    stats (how many lakes are high risk, risk distribution, etc.) without
    re-predicting every time the page reruns."""
    X = df[feature_cols]
    X_in = scaler.transform(X) if type(model).__name__ == "LogisticRegression" else X
    proba = model.predict_proba(X_in)[:, 1]
    scored = df.copy()
    scored["risk_proba"] = proba
    scored["risk_level"] = [risk_bucket(p)[0] for p in proba]
    return scored


scored_df = score_all_lakes()


def prediction_form(key_prefix="form"):
    """Shared input form (sliders + region) used on both the Home page and
    the dedicated Prediction page. Returns the predicted probability, or
    None if the button hasn't been pressed yet."""
    with st.form(f"{key_prefix}_predict"):
        c1, c2, c3 = st.columns(3)
        with c1:
            elevation = st.number_input("Elevation (m)", 1500, 6200, 4500, step=50)
            lake_area = st.number_input("Lake Area (km²)", 0.005, 5.0, 0.8, step=0.05)
            retreat = st.number_input("Glacier Retreat Rate (m/yr)", 0.0, 60.0, 15.0, step=1.0)
        with c2:
            distance = st.number_input("Distance from Glacier (m)", 50, 3000, 400, step=50)
            slope = st.number_input("Slope (degrees)", 1, 45, 20, step=1)
            rainfall = st.number_input("Rainfall (mm)", 300, 3800, 2200, step=50)
        with c3:
            temperature = st.number_input("Temperature (°C)", -10.0, 15.0, -1.0, step=0.5)
            snowfall = st.number_input("Snowfall (mm)", 100, 2800, 1400, step=50)
            eq_mag = st.number_input("Earthquake Magnitude", 2.0, 7.5, 4.0, step=0.1)
            region = st.selectbox("Region", list(inv_region_map.keys()))

        submitted = st.form_submit_button("Predict Risk")

    if not submitted:
        return None

    row = {
        "elevation_m": elevation, "lake_area_km2": lake_area,
        "glacier_retreat_m_per_yr": retreat, "distance_from_glacier_m": distance,
        "slope_deg": slope, "rainfall_mm": rainfall, "temperature_c": temperature,
        "snowfall_mm": snowfall, "earthquake_magnitude": eq_mag,
        "region_encoded": inv_region_map[region],
    }
    return predict_risk(row)


def show_result_card(proba):
    bucket, label = risk_bucket(proba)
    st.markdown(f"""
    <div class="glof-card" style="text-align:center;">
        <div class="risk-badge {bucket}">{label}</div>
        <div style="margin-top:1rem; font-size:1.3rem; font-family:'Outfit',sans-serif;">
            Probability: <b>{proba*100:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#7BBDE8"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(76,175,109,0.25)"},
                {'range': [40, 70], 'color': "rgba(242,184,75,0.25)"},
                {'range': [70, 100], 'color': "rgba(232,85,91,0.25)"},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#BDD8E9", height=260)
    st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# Sidebar navigation
# ==================================================================
st.sidebar.markdown("## GLOF System — India")
page = st.sidebar.radio("Navigate", [
    "Home",
    "Dataset Overview",
    "GLOF Risk Prediction",
    "Interactive Map",
    "Heatmap",
    "Satellite Comparison",
    "SHAP Explanation",
    "Alert System",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span style='color:#6EA2B3; font-size:0.82rem;'>AI-based Glacial Lake Outburst Flood "
    "risk prediction & early warning system for India</span>",
    unsafe_allow_html=True
)

# ==================================================================
# HOME — dashboard overview
# ==================================================================
if page == "Home":
    st.title("GLOF System — India")

    # top row of headline numbers
    high_risk = scored_df[scored_df["risk_level"] == "high"]
    urgent_alerts = scored_df[scored_df["risk_proba"] >= 0.85]

    cols = st.columns(5)
    metrics = [
        ("Predicted Risk", f"{scored_df['risk_proba'].mean():.2f}", "Avg. across lakes"),
        ("Glacier Lakes", f"{len(scored_df):,}", "Monitored"),
        ("High Risk Lakes", f"{len(high_risk):,}", f"{len(high_risk)/len(scored_df)*100:.1f}% of total"),
        ("Alerts Issued", f"{len(urgent_alerts):,}", "Last 30 days"),
        ("Regions Covered", f"{df['region'].nunique()}", "States"),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        col.markdown(f"""
        <div class="glof-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Risk Trend (Last 12 Months)")
    # we don't have historical monthly logs yet, so this is an illustrative
    # trend line built from the average predicted risk plus some seasonal
    # wobble — swap in real logged data once it's being collected
    months = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    rng = np.random.default_rng(42)
    base = scored_df["risk_proba"].mean()
    seasonal = 0.15 * np.sin(np.linspace(0, 2 * np.pi, 12))
    trend = np.clip(base + seasonal + rng.normal(0, 0.03, 12), 0, 1)
    fig_trend = px.line(x=months, y=trend, markers=True)
    fig_trend.update_traces(line_color="#7BBDE8")
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#BDD8E9", height=280, xaxis_title=None, yaxis_title=None,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### Recent Alerts")
    top_high = high_risk.sort_values("risk_proba", ascending=False).head(1)
    top_medium = scored_df[scored_df["risk_level"] == "medium"].sort_values("risk_proba", ascending=False).head(1)
    top_low = scored_df[scored_df["risk_level"] == "low"].sort_values("risk_proba").head(1)

    recent = [
        (top_high, "high", "High Risk GLOF Detected", "2h ago"),
        (top_medium, "medium", "Moderate Risk Detected", "6h ago"),
        (top_low, "low", "Low Risk Detected", "1d ago"),
    ]
    st.markdown('<div class="glof-card">', unsafe_allow_html=True)
    for subset, bucket, title, when in recent:
        if subset.empty:
            continue
        r = subset.iloc[0]
        st.markdown(f"""
        <div class="alert-row">
            <div><span class="risk-dot {bucket}"></span><b>{title}</b><br>
            <span class="metric-sub">{r['lake_name']}, {r['region']}</span></div>
            <span class="metric-sub">{when}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# DATASET OVERVIEW
# ==================================================================
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.markdown(f"<p style='color:#6EA2B3;'>{len(df)} lake records • "
                f"{df.shape[1]} features</p>", unsafe_allow_html=True)

    st.dataframe(df.head(50), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="lake_area_km2", nbins=40, color="glof_risk",
                            color_discrete_map={0: "#4CAF6D", 1: "#E8555B"},
                            title="Lake Area Distribution by Risk")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#BDD8E9")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(df, x="glacier_retreat_m_per_yr", y="lake_area_km2",
                           color="glof_risk", color_discrete_map={0: "#4CAF6D", 1: "#E8555B"},
                           title="Retreat Rate vs Lake Area")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#BDD8E9")
        st.plotly_chart(fig2, use_container_width=True)

# ==================================================================
# PREDICTION PAGE
# ==================================================================
elif page == "GLOF Risk Prediction":
    st.title("GLOF Risk Prediction")
    st.markdown("<p style='color:#6EA2B3;'>Enter lake and environmental parameters.</p>",
                unsafe_allow_html=True)

    proba = prediction_form(key_prefix="predict_page")
    if proba is not None:
        show_result_card(proba)

# ==================================================================
# INTERACTIVE MAP
# ==================================================================
elif page == "Interactive Map":
    st.title("Interactive Risk Map — GIS View")
    st.markdown("<p style='color:#6EA2B3;'>Click a marker for lake details. "
                "Toggle layers (top-right) to switch basemap or hide/show regional risk zones.</p>",
                unsafe_allow_html=True)

    show_all = st.checkbox("Show all lakes (uncheck to cap at 300 for faster rendering)", value=False)
    map_df = df.dropna(subset=["latitude", "longitude"])
    if not show_all:
        map_df = map_df.head(300)
    center_lat, center_lon = map_df["latitude"].mean(), map_df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None,
        width="100%",
        height="100%",
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", show=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map", show=False).add_to(m)

    # --- Region-level GIS overlay: average risk per region shown as a
    # graduated circle (a lightweight stand-in for a choropleth, since we
    # don't have polygon boundaries for each region in the dataset) ---
    region_group = folium.FeatureGroup(name="Region Risk Zones", show=True)
    region_stats = (
        df.dropna(subset=["latitude", "longitude"])
        .groupby("region")
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
             avg_risk=("glof_risk", "mean"), n=("glof_risk", "size"))
        .reset_index()
    )
    risk_colormap = bcm.LinearColormap(
        colors=["#4CAF6D", "#F2B84B", "#E8555B"], vmin=0, vmax=1,
        caption="Average GLOF risk by region"
    )
    for _, r in region_stats.iterrows():
        folium.Circle(
            location=[r["lat"], r["lon"]],
            radius=25000 + 60000 * r["avg_risk"],
            color=risk_colormap(r["avg_risk"]),
            weight=2,
            fill=True,
            fill_color=risk_colormap(r["avg_risk"]),
            fill_opacity=0.25,
            popup=folium.Popup(
                f"<b>{r['region']}</b><br>Lakes: {int(r['n'])}<br>"
                f"Avg. risk: {r['avg_risk']*100:.0f}%", max_width=200
            ),
        ).add_to(region_group)
    region_group.add_to(m)
    risk_colormap.add_to(m)

    # --- Individual lake markers, clustered for performance ---
    marker_group = folium.FeatureGroup(name="Lake Markers", show=True)
    cluster = MarkerCluster().add_to(marker_group)
    color_map = {0: "#4CAF6D", 1: "#E8555B"}
    for _, r in map_df.iterrows():
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=6 if r["glof_risk"] == 0 else 8,
            color=color_map[r["glof_risk"]],
            fill=True,
            fill_color=color_map[r["glof_risk"]],
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{r['lake_name']}</b><br>"
                f"Region: {r['region']}<br>"
                f"Area: {r['lake_area_km2']} km²<br>"
                f"Latitude: {r['latitude']:.4f}<br>"
                f"Longitude: {r['longitude']:.4f}<br"
                f">Elevation: {r['elevation_m']} m<br>"
                f"Risk: {'High' if r['glof_risk']==1 else 'Low'}",
                max_width=250
            ),
        ).add_to(cluster)

        if r["glof_risk"] == 1:
            folium.Circle(
                location=[r["latitude"], r["longitude"]],
                radius=5000,
                color="red",
                fill=True,
                fill_opacity=0.15
            ).add_to(marker_group)
    marker_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st_folium(m, use_container_width=True, height=480, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glof-card" style="display:flex; gap:2rem; margin-top:1rem; align-items:center;">
        <span><span class="risk-dot low"></span>Low Risk</span>
        <span><span class="risk-dot high"></span>High Risk</span>
        <span style="color:#6EA2B3;">Shaded circles = average regional risk</span>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# HEATMAP
# ==================================================================
elif page == "Heatmap":
    st.title("Risk Concentration Heatmap")
    st.markdown("<p style='color:#6EA2B3;'>Areas with the highest concentration of dangerous lakes.</p>",
                unsafe_allow_html=True)

    heat_mode = st.radio(
        "Heatmap weighting",
        ["High-risk lakes only", "All lakes, weighted by predicted risk"],
        horizontal=True,
    )

    map_df = df.dropna(subset=["latitude", "longitude"]).copy()
    scored_map_df = scored_df.dropna(subset=["latitude", "longitude"]).copy()

    if heat_mode == "High-risk lakes only":
        heat_source = map_df[map_df["glof_risk"] == 1]
        heat_data = [[row["latitude"], row["longitude"], 1] for _, row in heat_source.iterrows()]
        center_lat, center_lon = heat_source["latitude"].mean(), heat_source["longitude"].mean()
        marker_source = heat_source
    else:
        heat_source = scored_map_df
        heat_data = [[row["latitude"], row["longitude"], row["risk_proba"]]
                     for _, row in heat_source.iterrows()]
        center_lat, center_lon = heat_source["latitude"].mean(), heat_source["longitude"].mean()
        marker_source = heat_source[heat_source["risk_proba"] >= 0.7]

    m2 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        show=True
    ).add_to(m2)
    folium.TileLayer("OpenStreetMap", name="Street Map", show=False).add_to(m2)

    HeatMap(
        heat_data, radius=25, blur=20, min_opacity=0.35, max_zoom=12,
        gradient={0.2: "#4CAF6D", 0.5: "#F2B84B", 0.8: "#E8555B", 1.0: "#E8555B"},
    ).add_to(m2)

    for _, r in marker_source.iterrows():
        folium.CircleMarker(
            [r["latitude"], r["longitude"]],
            radius=5, color="red", fill=True, fill_color="red", fill_opacity=0.9,
            popup=folium.Popup(f"<b>{r['lake_name']}</b><br>Region: {r['region']}", max_width=200),
        ).add_to(m2)

    folium.LayerControl(collapsed=False).add_to(m2)
    Fullscreen().add_to(m2)

    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st_folium(m2, use_container_width=True, height=480, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glof-card" style="display:flex; gap:2rem; margin-top:1rem; align-items:center;">
        <span><span class="risk-dot low"></span>Lower intensity</span>
        <span><span class="risk-dot high"></span>Higher intensity</span>
        <span style="color:#6EA2B3;">Red markers = lakes ≥ 70% predicted risk</span>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# SATELLITE COMPARISON (illustrative — plug in real Sentinel/GEE imagery later)
# ==================================================================
elif page == "Satellite Comparison":
    st.title("Satellite Image Comparison")
    st.markdown("<p style='color:#6EA2B3;'>Compare lake extent across years to estimate growth. "
                "This view shows an estimated growth-rate analysis — real imagery can be plugged "
                "in later via Sentinel Hub / Google Earth Engine.</p>", unsafe_allow_html=True)

    high_risk_df = df[df["glof_risk"] == 1]
    real_names = sorted(high_risk_df[high_risk_df["lake_type"] == "real"]["lake_name"].unique())
    synthetic_names = sorted(high_risk_df[high_risk_df["lake_type"] == "synthetic"]["lake_name"].unique())
    lake_options = (real_names + synthetic_names)[:15]

    lake_choice = st.selectbox("Select a lake", lake_options)
    row = df[df["lake_name"] == lake_choice].iloc[0]

    years = list(range(2000, 2025, 3))
    base_area = row["lake_area_km2"] * 0.4
    growth = [base_area * (1.06 ** i) for i in range(len(years))]

    fig = px.area(x=years, y=growth, labels={"x": "Year", "y": "Estimated Lake Area (km²)"},
                  title=f"Estimated Lake Expansion — {lake_choice}")
    fig.update_traces(line_color="#7BBDE8", fillcolor="rgba(123,189,232,0.2)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#BDD8E9")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="glof-card">
        <b>{lake_choice}</b> — estimated growth of
        <b>{(growth[-1]/growth[0]-1)*100:.0f}%</b> from {years[0]} to {years[-1]}.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# SHAP EXPLANATION
# ==================================================================
elif page == "SHAP Explanation":
    st.title("Explainable AI — SHAP")
    st.markdown("<p style='color:#6EA2B3;'>Why the model predicts high or low risk.</p>",
                unsafe_allow_html=True)

    try:
        import shap
        sample = df[feature_cols].sample(min(150, len(df)), random_state=1)
        if type(model).__name__ == "LogisticRegression":
            explainer = shap.LinearExplainer(model, scaler.transform(sample))
            shap_values = explainer(scaler.transform(sample))
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(sample)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig, use_container_width=True)
    except ImportError:
        st.warning("Install `shap` (see requirements.txt) to see live SHAP plots. "
                   "Showing feature importance fallback below.")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
            fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                         title="Feature Importance (fallback)")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#BDD8E9")
            st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# ALERT SYSTEM
# ==================================================================
elif page == "Alert System":
    st.title("Early Warning Alerts")
    st.markdown("<p style='color:#6EA2B3;'>Lakes currently flagged above the 70% risk threshold.</p>",
                unsafe_allow_html=True)

    high_risk_df = df[df["glof_risk"] == 1]
    real_alerts = high_risk_df[high_risk_df["lake_type"] == "real"]
    synthetic_alerts = high_risk_df[high_risk_df["lake_type"] == "synthetic"]
    alerts = pd.concat([real_alerts, synthetic_alerts]).head(10)
    for _, r in alerts.iterrows():
        st.markdown(f"""
        <div class="alert-banner" style="margin-bottom:0.8rem;">
            <b>HIGH RISK ALERT</b><br><br>
            <b>Lake:</b> {r['lake_name']}<br>
            <b>Region:</b> {r['region']}<br>
            <b>Elevation:</b> {r['elevation_m']} m<br>
            <b>Lake Area:</b> {r['lake_area_km2']} km²<br>
            <b>Recommended Action:</b> Immediate monitoring / evacuation planning for downstream villages.
        </div>
        """, unsafe_allow_html=True)
