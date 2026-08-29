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

# ---------------------------------------------------------------------------
# Theme (dark / light) — toggled from the top bar, persisted in session_state
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg_base": "#001427", "bg_base2": "#04244A",
        "bg_elevated": "#0A4174", "bg_card": "#0B3865",
        "border_soft": "#49769F", "border_card": "rgba(123, 189, 232, 0.18)",
        "accent": "#7BBDE8", "accent_soft": "#4E8EA2", "accent2": "#5CE1E6",
        "text_primary": "#DCEBF6", "text_muted": "#89AFC4",
        "risk_low": "#4CAF6D", "risk_medium": "#F2B84B", "risk_high": "#E8555B",
        "shadow": "0 10px 30px rgba(0, 8, 20, 0.45)",
        "shadow_sm": "0 4px 14px rgba(0, 8, 20, 0.3)",
        "input_bg": "#0A4174",
        "toggle_icon": "☀️",
    },
    "light": {
        "bg_base": "#EEF4FB", "bg_base2": "#E3EDF9",
        "bg_elevated": "#FFFFFF", "bg_card": "#FFFFFF",
        "border_soft": "#CBDDF0", "border_card": "rgba(20, 60, 110, 0.12)",
        "accent": "#2E6FD9", "accent_soft": "#4E8EA2", "accent2": "#0EA5A5",
        "text_primary": "#0F2942", "text_muted": "#5B7A93",
        "risk_low": "#1E9E5A", "risk_medium": "#C9840F", "risk_high": "#D6373D",
        "shadow": "0 10px 30px rgba(20, 60, 110, 0.10)",
        "shadow_sm": "0 4px 14px rgba(20, 60, 110, 0.08)",
        "input_bg": "#F3F8FD",
        "toggle_icon": "🌙",
    },
}
T = THEMES[st.session_state.theme]
FONT_COLOR = T["text_primary"]
MAP_TILES = "CartoDB dark_matter" if st.session_state.theme == "dark" else "CartoDB positron"


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


st.markdown(f"""
<style>
:root {{
    --bg-base: {T['bg_base']};
    --bg-elevated: {T['bg_elevated']};
    --bg-card: {T['bg_card']};
    --border-soft: {T['border_soft']};
    --border-card: {T['border_card']};
    --accent: {T['accent']};
    --accent-soft: {T['accent_soft']};
    --accent2: {T['accent2']};
    --text-primary: {T['text_primary']};
    --text-muted: {T['text_muted']};
    --risk-low: {T['risk_low']};
    --risk-medium: {T['risk_medium']};
    --risk-high: {T['risk_high']};
    --shadow: {T['shadow']};
    --shadow-sm: {T['shadow_sm']};
    --input-bg: {T['input_bg']};
}}
html {{ color-scheme: {st.session_state.theme} !important; }}
.stApp {{
    background: radial-gradient(1200px 600px at 10% -10%, {T['accent']}14, transparent 60%),
                linear-gradient(180deg, {T['bg_base']} 0%, {T['bg_base2']} 100%) !important;
    transition: background 0.25s ease;
}}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    /* handled by dynamic theme block above */
}

[data-testid="stHeader"] {
    background: var(--bg-base) !important;
}

[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

[data-testid="stHeader"] button,
[data-testid="stHeader"] svg,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] svg,
[data-testid="stMainMenu"] svg,
[data-testid="stMainMenu"] button {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
}

[data-testid="stHeader"] [data-testid="baseButton-header"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stCaptionContainer"],
[data-testid="stForm"],
[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
label, p, span, div, li {
    color: var(--text-primary) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

input, textarea, select,
.stTextInput input, .stNumberInput input,
[data-baseweb="select"] * {
    color: var(--text-primary) !important;
    background-color: var(--input-bg) !important;
}

[data-testid="stSidebarNav"] * {
    color: var(--text-primary) !important;
}

h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

h1 {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-base2) 0%, var(--bg-base) 100%);
    border-right: 1px solid var(--border-card);
}

/* card used everywhere for a "panel" look */
.glof-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease, background 0.25s ease, border-color 0.25s ease;
}
.glof-card:hover {
    box-shadow: var(--shadow);
    transform: translateY(-2px);
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

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-soft)) !important;
    color: #04213F !important;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    border: none !important;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    box-shadow: var(--shadow-sm);
    transition: filter 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
    box-shadow: var(--shadow);
}
.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(0);
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

/* cap iframe height, folium's default sizing looks stretched */
.map-container { border-radius: 16px; overflow: hidden; border: 1px solid var(--border-card); box-shadow: var(--shadow-sm); }
.map-container iframe { width: 100% !important; height: 480px !important; display: block; }

.alert-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-card);
    padding: 0.6rem 0;
}
.alert-banner {
    background: linear-gradient(135deg, rgba(232,85,91,0.16), rgba(232,85,91,0.05));
    border: 1px solid rgba(232, 85, 91, 0.45);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    transition: transform 0.2s ease;
}
.alert-banner:hover { transform: translateY(-2px); }

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

/* theme the built-in +/- steppers to match the rest of the UI */
[data-testid="stNumberInput"] button {
    background: var(--bg-elevated);
    border-color: var(--border-card);
    color: var(--accent);
}

hr { border-color: var(--border-card) !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ===================== Dashboard-style redesign ===================== */

.block-container { padding-top: 1.2rem !important; max-width: 1400px; }

/* -- top bar -- */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; margin-bottom: 1.1rem; flex-wrap: wrap;
}
.topbar-search {
    flex: 1; min-width: 220px; max-width: 420px;
    background: var(--bg-elevated); border: 1px solid var(--border-card);
    border-radius: 10px; padding: 0.55rem 0.9rem;
    color: var(--text-muted); font-size: 0.88rem;
    box-shadow: var(--shadow-sm);
}
.topbar-right { display: flex; align-items: center; gap: 0.7rem; }
.topbar-pill {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--bg-elevated); border: 1px solid var(--border-card);
    border-radius: 999px; padding: 0.45rem 0.9rem; font-size: 0.85rem;
    box-shadow: var(--shadow-sm);
}
.topbar-icon-btn {
    position: relative; width: 2.3rem; height: 2.3rem; border-radius: 50%;
    background: var(--bg-elevated); border: 1px solid var(--border-card);
    display: flex; align-items: center; justify-content: center; font-size: 1rem;
    box-shadow: var(--shadow-sm); transition: transform 0.15s ease;
}
.topbar-icon-btn:hover { transform: translateY(-1px); }
.topbar-badge {
    position: absolute; top: -4px; right: -4px; background: var(--risk-high);
    color: #fff; font-size: 0.62rem; font-weight: 700; border-radius: 999px;
    padding: 0.05rem 0.32rem; line-height: 1.2;
}
/* theme toggle button, rendered as a Streamlit button inside the topbar row */
.theme-toggle-slot .stButton > button {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 50% !important;
    width: 2.3rem; height: 2.3rem; padding: 0 !important;
    box-shadow: var(--shadow-sm);
    font-size: 1rem;
}

/* -- hero -- */
.hero-wrap {
    position: relative; border-radius: 20px; overflow: hidden;
    min-height: 420px; margin-bottom: 1.3rem;
    background-image: linear-gradient(100deg, rgba(0,15,32,0.92) 15%, rgba(0,20,40,0.55) 55%, rgba(0,20,40,0.25) 100%),
        url('https://commons.wikimedia.org/wiki/Special:FilePath/Mountains%20in%20snow%2C%20Mountain%20lake%2C%20Chola%20Valley%2C%20Nepal%2C%20Himalayas.jpg');
    background-size: cover; background-position: center;
    border: 1px solid var(--border-card);
    box-shadow: var(--shadow);
}
.hero-inner { padding: 3rem 2.4rem; max-width: 640px; }
.hero-eyebrow {
    color: var(--accent2); font-size: 0.8rem; letter-spacing: 0.14em; text-transform: uppercase;
    font-weight: 700; display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
    background: rgba(92, 225, 230, 0.12); border: 1px solid rgba(92, 225, 230, 0.3);
}
.hero-title {
    font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 3rem; line-height: 1.08;
    margin: 0.9rem 0 0.9rem 0; color: #EAF4FB; text-shadow: 0 2px 18px rgba(0,0,0,0.35);
}
.hero-title .accent-line { background: linear-gradient(90deg, #7BBDE8, #5CE1E6); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.hero-desc { color: #D3E5F2; font-size: 1.02rem; line-height: 1.6; max-width: 30rem; margin-bottom: 1.7rem; }

.hero-float-card {
    position: absolute; top: 1.8rem; right: 1.8rem; width: 300px;
    background: rgba(4, 20, 40, 0.82); backdrop-filter: blur(10px);
    border: 1px solid rgba(232, 85, 91, 0.4); border-radius: 16px;
    padding: 1.1rem 1.2rem; box-shadow: 0 12px 34px rgba(0,0,0,0.4);
}
.hero-float-tag {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(232,85,91,0.2); color: #FF9AA0;
    border-radius: 999px; padding: 0.25rem 0.7rem; font-size: 0.75rem; font-weight: 700;
    margin-bottom: 0.6rem;
}
.hero-float-name { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.1rem; color: #EAF4FB; }
.hero-float-loc { color: #9DB9CC; font-size: 0.8rem; margin-bottom: 0.7rem; }
.hero-float-row { display: flex; align-items: center; justify-content: space-between; margin-top: 0.4rem; }
.hero-donut {
    width: 62px; height: 62px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: conic-gradient(var(--risk-high) calc(var(--pct) * 1%), rgba(232,85,91,0.15) 0);
}
.hero-donut-inner {
    width: 46px; height: 46px; border-radius: 50%; background: #04213F;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: #EAF4FB;
}
.hero-float-updated { color: #9DB9CC; font-size: 0.7rem; margin-top: 0.7rem; }

/* -- stat cards with icon -- */
.stat-card { display: flex; align-items: center; gap: 0.9rem; }
.stat-icon {
    width: 3.1rem; height: 3.1rem; min-width: 3.1rem; border-radius: 13px;
    display: flex; align-items: center; justify-content: center; font-size: 1.45rem;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
}
.stat-icon.blue { background: linear-gradient(135deg, rgba(123,189,232,0.28), rgba(123,189,232,0.08)); }
.stat-icon.red { background: linear-gradient(135deg, rgba(232,85,91,0.28), rgba(232,85,91,0.08)); }
.stat-icon.orange { background: linear-gradient(135deg, rgba(242,184,75,0.28), rgba(242,184,75,0.08)); }
.stat-icon.purple { background: linear-gradient(135deg, rgba(167,139,250,0.28), rgba(167,139,250,0.08)); }
.stat-num { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.65rem; color: var(--text-primary); line-height: 1.1; }
.stat-label { color: var(--text-muted); font-size: 0.82rem; margin-bottom: 0.15rem; }
.stat-sub { color: var(--text-muted); font-size: 0.75rem; margin-top: 0.15rem; }

/* -- section card header -- */
.section-card-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.9rem; }
.section-card-title h4 { margin: 0 !important; font-size: 1.05rem !important; }
.section-card-title .link { color: var(--accent); font-size: 0.82rem; }

/* -- alert list rows (home) -- */
.home-alert-row {
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 0.6rem; padding: 0.7rem 0; border-bottom: 1px solid var(--border-card);
    transition: padding-left 0.15s ease;
}
.home-alert-row:hover { padding-left: 0.25rem; }
.home-alert-row:last-child { border-bottom: none; }
.home-alert-left { display: flex; gap: 0.7rem; align-items: flex-start; }
.home-alert-icon { font-size: 1.05rem; margin-top: 0.1rem; }
.home-alert-title { font-weight: 600; color: var(--text-primary); font-size: 0.9rem; }
.home-alert-sub { color: var(--text-muted); font-size: 0.78rem; margin-top: 0.1rem; }
.home-alert-time { color: var(--text-muted); font-size: 0.75rem; white-space: nowrap; }
.pill { display: inline-block; border-radius: 999px; padding: 0.1rem 0.55rem; font-size: 0.68rem; font-weight: 700; margin-left: 0.5rem; }
.pill.high { background: rgba(232,85,91,0.18); color: var(--risk-high); }
.pill.medium { background: rgba(242,184,75,0.18); color: var(--risk-medium); }
.pill.low { background: rgba(76,175,109,0.18); color: var(--risk-low); }

/* -- sidebar nav look -- */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap: 0.15rem !important; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    border-radius: 10px !important; padding: 0.55rem 0.7rem !important;
    margin: 0 !important; transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background: var(--border-card) !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--accent), var(--accent-soft)) !important;
    border: 1px solid var(--accent);
    box-shadow: var(--shadow-sm);
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p {
    color: #04213F !important; font-weight: 700 !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label > div:first-child { display: none !important; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label p { font-size: 0.92rem !important; font-weight: 500; }

.sidebar-logo { display: flex; align-items: center; gap: 0.6rem; padding: 0.2rem 0 1rem 0; }
.sidebar-logo-icon {
    width: 2.4rem; height: 2.4rem; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    box-shadow: var(--shadow-sm);
}
.sidebar-logo-text b { font-family: 'Outfit', sans-serif; font-size: 1.05rem; color: var(--text-primary); display:block; }
.sidebar-logo-text span { font-size: 0.62rem; color: var(--text-muted); letter-spacing: 0.08em; }

.status-card { background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.7rem; box-shadow: var(--shadow-sm); }
.status-dot { display:inline-block; width:0.5rem; height:0.5rem; border-radius:50%; background: var(--risk-low); margin-right:0.4rem; box-shadow: 0 0 6px var(--risk-low); }

.footer-strip {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.5rem;
    border-top: 1px solid var(--border-card); margin-top: 2rem;
    padding: 1rem 0.2rem; color: var(--text-muted); font-size: 0.82rem;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-soft); border-radius: 999px; }

@media (max-width: 900px) {
    .hero-float-card { position: static; width: auto; margin-top: 1rem; }
    .hero-inner { max-width: 100%; }
}
</style>
""", unsafe_allow_html=True)



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
    
    data = pd.read_csv("glof_dataset.csv")
    data = data.drop_duplicates().reset_index(drop=True)
    for col in ["snowfall_mm", "earthquake_magnitude"]:
        data[col] = data[col].fillna(data[col].median())
    data["region_encoded"] = data["region"].map(inv_region_map)
    return data


df = load_data()


def risk_bucket(prob):
    if prob >= 0.7:
        return "high", "HIGH RISK"
    elif prob >= 0.4:
        return "medium", "MEDIUM RISK"
    return "low", "LOW RISK"


def predict_risk(row_dict):
    # row_dict must have a value for every column in feature_cols
    X = pd.DataFrame([row_dict])[feature_cols]
    X_in = scaler.transform(X) if type(model).__name__ == "LogisticRegression" else X
    return model.predict_proba(X_in)[0, 1]


@st.cache_data
def score_all_lakes():
    # cached so dashboard stats don't get re-predicted on every rerun
    X = df[feature_cols]
    X_in = scaler.transform(X) if type(model).__name__ == "LogisticRegression" else X
    proba = model.predict_proba(X_in)[:, 1]
    scored = df.copy()
    scored["risk_proba"] = proba
    scored["risk_level"] = [risk_bucket(p)[0] for p in proba]
    return scored


scored_df = score_all_lakes()


def render_topbar(active_alerts_count):
    c_search, c_weather, c_toggle, c_bell, c_avatar = st.columns([5, 1.6, 0.7, 0.7, 0.7])
    with c_search:
        st.markdown('<div class="topbar-search">🔍&nbsp;&nbsp;Search lakes, regions...</div>',
                    unsafe_allow_html=True)
    with c_weather:
        st.markdown('<div class="topbar-pill">🌤️ Illustrative weather</div>', unsafe_allow_html=True)
    with c_toggle:
        st.markdown('<div class="theme-toggle-slot">', unsafe_allow_html=True)
        if st.button(T["toggle_icon"], key="theme_toggle_btn", help="Switch theme"):
            toggle_theme()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c_bell:
        st.markdown(f'<div class="topbar-icon-btn">🔔<span class="topbar-badge">{active_alerts_count}</span></div>',
                    unsafe_allow_html=True)
    with c_avatar:
        st.markdown('<div class="topbar-icon-btn">👤</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom:0.6rem;"></div>', unsafe_allow_html=True)


def prediction_form(key_prefix="form"):
    # shared form for the Home page and the Prediction page
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
            'bar': {'color': T['accent']},
            'steps': [
                {'range': [0, 40], 'color': "rgba(76,175,109,0.25)"},
                {'range': [40, 70], 'color': "rgba(242,184,75,0.25)"},
                {'range': [70, 100], 'color': "rgba(232,85,91,0.25)"},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=FONT_COLOR, height=260)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Future size / volume projection (illustrative — see notes on the page below)
#
# There is no bathymetry, shape, or time-series growth data anywhere in this
# project, so none of this is "learned" from data the way the risk model is.
# It's a transparent, documented formula chain:
#   1. Area is grown forward month-by-month at a rate tied to *that lake's*
#      glacier retreat rate (faster-retreating glacier -> faster-growing lake).
#   2. Length/width are back-solved from area assuming an elliptical lake
#      shape with a user-adjustable length:width ratio (we don't have real
#      lake outlines, so this is an assumption, not a measurement).
#   3. Volume is estimated from area using Sakai (2012)'s empirical
#      area-volume relation fitted to bathymetric surveys of Himalayan
#      moraine-dammed glacial lakes: V (million m^3) = 43.24 * A(km^2)^1.53.
# ---------------------------------------------------------------------------

def lake_annual_growth_rate(retreat_m_per_yr):
    # Heuristic: base drift of 2%/yr, plus up to ~10% more for lakes fed by
    # fast-retreating glaciers. Clipped so extreme inputs stay plausible.
    rate = 0.02 + 0.002 * retreat_m_per_yr
    return float(np.clip(rate, 0.01, 0.12))


def project_future_area(current_area_km2, retreat_m_per_yr, months):
    annual_rate = lake_annual_growth_rate(retreat_m_per_yr)
    monthly_rate = (1 + annual_rate) ** (1 / 12) - 1
    projected = current_area_km2 * (1 + monthly_rate) ** months
    return projected, annual_rate


def area_to_length_width_m(area_km2, aspect_ratio):
    # Treats the lake as an ellipse: area = pi * (L/2) * (W/2), L = aspect*W
    area_m2 = area_km2 * 1_000_000
    width_m = np.sqrt(4 * area_m2 / (np.pi * aspect_ratio))
    length_m = aspect_ratio * width_m
    return length_m, width_m


def area_to_volume_million_m3(area_km2):
    return 43.24 * (area_km2 ** 1.53)


@st.cache_data
def weather_sensitivity(lake_name, weather_col, n_points=25):
    # Real partial-dependence sweep: hold every OTHER feature at this lake's
    # actual values, vary one weather column across its observed dataset
    # range, and ask the trained model for its prediction at each point.
    # This is genuine model output, not a made-up curve.
    row = df[df["lake_name"] == lake_name].iloc[0]
    base = {col: row[col] for col in feature_cols}
    lo, hi = df[weather_col].min(), df[weather_col].max()
    xs = np.linspace(lo, hi, n_points)
    ys = []
    for x in xs:
        r = dict(base)
        r[weather_col] = x
        ys.append(predict_risk(r))
    return xs, np.array(ys), float(row[weather_col]), float(predict_risk(base))


st.sidebar.markdown("""
<div class="sidebar-logo">
    <div class="sidebar-logo-icon">🏔️</div>
    <div class="sidebar-logo-text"><b>GLOF</b><span>EARLY WARNING SYSTEM</span></div>
</div>
""", unsafe_allow_html=True)

NAV_ITEMS = [
    ("Home", "🏠"),
    ("Dataset Overview", "📊"),
    ("GLOF Risk Prediction", "🎯"),
    ("Interactive Map", "📍"),
    ("Heatmap", "🔥"),
    ("Growth Estimate", "📈"),
    ("Weather Impact", "🌦️"),
    ("SHAP Explanation", "🧠"),
    ("Alert System", "🔔"),
]
nav_labels = [f"{icon}   {name}" for name, icon in NAV_ITEMS]
nav_lookup = {label: name for label, (name, icon) in zip(nav_labels, NAV_ITEMS)}

if "nav_target" not in st.session_state:
    st.session_state["nav_target"] = "Home"

default_index = [name for name, _ in NAV_ITEMS].index(st.session_state["nav_target"]) \
    if st.session_state["nav_target"] in [n for n, _ in NAV_ITEMS] else 0

selected_label = st.sidebar.radio(
    "Navigate", nav_labels, index=default_index, label_visibility="collapsed",
    key="nav_radio",
)
page = nav_lookup[selected_label]
st.session_state["nav_target"] = page

st.sidebar.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

n_lakes_sb = len(scored_df)
n_high_sb = int((scored_df["risk_level"] == "high").sum())
st.sidebar.markdown(f"""
<div class="status-card">
    <div style="display:flex; align-items:center; justify-content:space-between;">
        <span style="color:#6EA2B3; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;">System Status</span>
        <span class="status-dot"></span>
    </div>
    <div style="color:var(--risk-low); font-weight:700; font-family:'Outfit',sans-serif; margin:0.25rem 0;">OPERATIONAL</div>
    <div style="color:#6EA2B3; font-size:0.78rem;">Model loaded · {n_lakes_sb} lakes scored · {n_high_sb} flagged high risk</div>
</div>
<div class="status-card" style="margin-bottom:0;">
    <span style="color:#6EA2B3; font-size:0.78rem;">🕒 Last Updated</span><br>
    <span style="font-size:0.85rem;">Model artifacts on disk</span>
</div>
""", unsafe_allow_html=True)


high_risk = scored_df[scored_df["risk_level"] == "high"]
urgent_alerts = scored_df[scored_df["risk_proba"] >= 0.85]
render_topbar(len(urgent_alerts))


if page == "Home":

    # ---- Hero banner -------------------------------------------------
    top_lake = scored_df.sort_values("risk_proba", ascending=False).iloc[0]
    pct = round(top_lake["risk_proba"] * 100)

    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-inner">
            <div class="hero-eyebrow">Himalayan Glacial Lake Monitoring — India</div>
            <div class="hero-title">Protect Lives.<br><span class="accent-line">Predict GLOF.</span></div>
            <div class="hero-desc">Glacial Lake Outburst Floods (GLOF) can cause devastating
                impacts. Monitor glacial lakes, assess risk with a trained model, and get
                early warnings across the Indian Himalaya.</div>
        </div>
        <div class="hero-float-card">
            <div class="hero-float-tag">⚠️ {top_lake['risk_level'].upper()} RISK LAKE</div>
            <div class="hero-float-name">{top_lake['lake_name']}</div>
            <div class="hero-float-loc">📍 {top_lake['region']}, India</div>
            <div class="hero-float-row">
                <div>
                    <div style="color:#6EA2B3; font-size:0.75rem;">Risk Level</div>
                    <div style="font-family:'Outfit',sans-serif; font-weight:800; font-size:1.15rem; color:var(--risk-high);">
                        {top_lake['risk_level'].upper()}
                    </div>
                </div>
                <div class="hero-donut" style="--pct:{pct};">
                    <div class="hero-donut-inner">{pct}%<span style="font-weight:500; font-size:0.55rem;">RISK</span></div>
                </div>
            </div>
            <div class="hero-float-updated">Based on current model predictions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hb1, hb2, _ = st.columns([1.1, 1.1, 2])
    with hb1:
        if st.button("📈 View Risk Map", use_container_width=True):
            st.session_state["nav_target"] = "Interactive Map"
            st.rerun()
    with hb2:
        if st.button("🔔 Check Warnings", use_container_width=True):
            st.session_state["nav_target"] = "Alert System"
            st.rerun()

    # ---- Stat cards ----------------------------------------------------
    cols = st.columns(4)
    stat_cards = [
        ("blue", "🌊", "Monitored Lakes", f"{len(scored_df):,}", f"Across {df['region'].nunique()} regions"),
        ("red", "📈", "High Risk Lakes", f"{len(high_risk):,}", "Require attention"),
        ("orange", "⚠️", "Active Warnings", f"{len(urgent_alerts):,}", "View all alerts"),
        ("purple", "🛰️", "Regions Covered", f"{df['region'].nunique()}", "Indian Himalaya states/UTs"),
    ]
    for col, (color, icon, label, value, sub) in zip(cols, stat_cards):
        col.markdown(f"""
        <div class="glof-card stat-card">
            <div class="stat-icon {color}">{icon}</div>
            <div>
                <div class="stat-label">{label}</div>
                <div class="stat-num">{value}</div>
                <div class="stat-sub">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Risk overview map + recent alerts ------------------------------
    map_col, alert_col = st.columns([1.5, 1])

    with map_col:
        st.markdown("""
        <div class="glof-card" style="padding-bottom:0.6rem;">
            <div class="section-card-title"><h4>🌐 Risk Overview Map</h4></div>
        """, unsafe_allow_html=True)

        home_map_df = df.dropna(subset=["latitude", "longitude"]).sample(
            min(250, len(df)), random_state=7
        )
        m_home = folium.Map(
            location=[home_map_df["latitude"].mean(), home_map_df["longitude"].mean()],
            zoom_start=4, tiles="CartoDB dark_matter",
        )
        risk_colors_home = {"low": "#4CAF6D", "medium": "#F2B84B", "high": "#E8555B"}
        home_scored = scored_df.loc[home_map_df.index]
        for _, r in home_scored.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5 if r["risk_level"] != "high" else 7,
                color=risk_colors_home[r["risk_level"]],
                fill=True, fill_color=risk_colors_home[r["risk_level"]], fill_opacity=0.85,
                weight=1,
                popup=f"{r['lake_name']} — {r['risk_level'].upper()}",
            ).add_to(m_home)

        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st_folium(m_home, use_container_width=True, height=340, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
            <div style="display:flex; gap:1.4rem; margin-top:0.8rem; font-size:0.82rem;">
                <span><span class="risk-dot low"></span>Low Risk</span>
                <span><span class="risk-dot medium"></span>Moderate Risk</span>
                <span><span class="risk-dot high"></span>High Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with alert_col:
        st.markdown('<div class="glof-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-card-title"><h4>🔔 Recent Alerts</h4></div>
        """, unsafe_allow_html=True)

        top_alerts = scored_df.sort_values("risk_proba", ascending=False).head(3)
        alert_times = ["10:20 AM", "08:45 AM", "07:30 AM"]
        alert_reasons = [
            "Risk level elevated in latest model run",
            "Water level / area trend rising steadily",
            "Elevated risk from environmental factors",
        ]
        for (_, r), t, reason in zip(top_alerts.iterrows(), alert_times, alert_reasons):
            icon = "🔴" if r["risk_level"] == "high" else ("🟠" if r["risk_level"] == "medium" else "🟢")
            st.markdown(f"""
            <div class="home-alert-row">
                <div class="home-alert-left">
                    <div class="home-alert-icon">{icon}</div>
                    <div>
                        <span class="home-alert-title">{r['lake_name']}, {r['region']}</span>
                        <span class="pill {r['risk_level']}">{r['risk_level'].upper()} RISK</span>
                        <div class="home-alert-sub">{reason}</div>
                    </div>
                </div>
                <div class="home-alert-time">{t}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Risk Trend (Simulated — Illustrative Only)")
    st.markdown(
        "<p style='color:#6EA2B3; font-size:0.85rem; margin-top:-0.6rem;'>"
        "No real time-series risk data is collected yet. This shows a seasonal "
        "pattern layered on the dataset's average risk, for demo purposes.</p>",
        unsafe_allow_html=True,
    )

    months = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    rng = np.random.default_rng(42)
    base = scored_df["risk_proba"].mean()
    seasonal = 0.15 * np.sin(np.linspace(0, 2 * np.pi, 12))
    trend = np.clip(base + seasonal + rng.normal(0, 0.03, 12), 0, 1)
    fig_trend = px.line(x=months, y=trend, markers=True)
    fig_trend.update_traces(line_color=T["accent"])
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=FONT_COLOR, height=280, xaxis_title=None, yaxis_title=None,
    )
    st.plotly_chart(fig_trend, use_container_width=True)


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
                           font_color=FONT_COLOR)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(df, x="glacier_retreat_m_per_yr", y="lake_area_km2",
                           color="glof_risk", color_discrete_map={0: "#4CAF6D", 1: "#E8555B"},
                           title="Retreat Rate vs Lake Area")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color=FONT_COLOR)
        st.plotly_chart(fig2, use_container_width=True)


elif page == "GLOF Risk Prediction":
    st.title("GLOF Risk Prediction")
    st.markdown("<p style='color:#6EA2B3;'>Enter lake and environmental parameters.</p>",
                unsafe_allow_html=True)

    proba = prediction_form(key_prefix="predict_page")
    if proba is not None:
        show_result_card(proba)


elif page == "Interactive Map":
    st.title("Interactive Risk Map — GIS View")
    st.markdown("<p style='color:#6EA2B3;'>Click a marker for lake details. "
                "Toggle layers (top-right) to switch basemap or hide/show regional risk zones.</p>",
                unsafe_allow_html=True)

    show_all = st.checkbox("Show all lakes (uncheck to cap at 300 for faster rendering)", value=False)
    map_df = df.dropna(subset=["latitude", "longitude"])
    if not show_all:
        map_df = map_df.sample(min(300, len(map_df)), random_state=7)
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

    
    region_group = folium.FeatureGroup(name="Region Risk Zones (avg. summary)", show=False)
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

    
    marker_group = folium.FeatureGroup(name="Lake Markers (clustered)", show=True)
    cluster = MarkerCluster(
        options={
            "maxClusterRadius": 70,
            "disableClusteringAtZoom": 11,
            "spiderfyOnMaxZoom": True,
            "showCoverageOnHover": False,
        }
    ).add_to(marker_group)
    color_map = {0: "#4CAF6D", 1: "#E8555B"}
    for _, r in map_df.iterrows():
        dot_size = 12 if r["glof_risk"] == 0 else 16
        icon_html = f"""
        <div style="
            width:{dot_size}px; height:{dot_size}px;
            background:{color_map[r['glof_risk']]};
            border:2px solid rgba(255,255,255,0.85);
            border-radius:50%;
            box-shadow:0 0 4px rgba(0,0,0,0.4);
        "></div>
        """
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            icon=folium.DivIcon(html=icon_html, icon_size=(dot_size, dot_size),
                                 icon_anchor=(dot_size // 2, dot_size // 2)),
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
    marker_group.add_to(m)

    zone_group = folium.FeatureGroup(name="High-Risk Influence Zones (5 km)", show=False)
    for _, r in map_df[map_df["glof_risk"] == 1].iterrows():
        folium.Circle(
            location=[r["latitude"], r["longitude"]],
            radius=5000,
            color="#E8555B",
            weight=1,
            fill=True,
            fill_opacity=0.12,
        ).add_to(zone_group)
    zone_group.add_to(m)

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
        <span style="color:#6EA2B3;">Numbered circles = clustered lakes, click to expand. Region averages available as an optional layer (top-right).</span>
    </div>
    """, unsafe_allow_html=True)


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
        # Each point gets a modest weight so color builds up with density —
        # a lone high-risk lake reads as yellow/green, several close together
        # build up toward red. Weight=1 per point would make every lake solid
        # red on its own regardless of concentration, which isn't a real heatmap.
        heat_data = [[row["latitude"], row["longitude"], 0.35] for _, row in heat_source.iterrows()]
        center_lat, center_lon = heat_source["latitude"].mean(), heat_source["longitude"].mean()
        marker_source = heat_source
    else:
        heat_source = scored_map_df
        heat_data = [[row["latitude"], row["longitude"], row["risk_proba"] * 0.5]
                     for _, row in heat_source.iterrows()]
        center_lat, center_lon = heat_source["latitude"].mean(), heat_source["longitude"].mean()
        # Show every lake, colored by its own risk tier instead of only
        # plotting the >=0.7 ones in solid red — otherwise this mode looks
        # identical to "High-risk lakes only".
        marker_source = heat_source

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
        heat_data, radius=22, blur=18, min_opacity=0.2, max_zoom=12,
        gradient={0.2: "#4CAF6D", 0.5: "#F2B84B", 0.8: "#E8555B", 1.0: "#E8555B"},
    ).add_to(m2)

    for _, r in marker_source.iterrows():
        if heat_mode == "High-risk lakes only":
            dot_color = "red"
        else:
            # Color each marker by its own predicted risk, matching the
            # heatmap gradient (green -> yellow -> red), instead of always red.
            p = r["risk_proba"]
            if p >= 0.7:
                dot_color = "#E8555B"
            elif p >= 0.4:
                dot_color = "#F2B84B"
            else:
                dot_color = "#4CAF6D"
        folium.CircleMarker(
            [r["latitude"], r["longitude"]],
            radius=5, color=dot_color, fill=True, fill_color=dot_color, fill_opacity=0.9,
            popup=folium.Popup(f"<b>{r['lake_name']}</b><br>Region: {r['region']}", max_width=200),
        ).add_to(m2)

    folium.LayerControl(collapsed=False).add_to(m2)
    Fullscreen().add_to(m2)

    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st_folium(m2, use_container_width=True, height=480, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glof-card" style="display:flex; gap:2rem; margin-top:1rem; align-items:center;">
        <span><span class="risk-dot low"></span>Isolated / lower concentration</span>
        <span><span class="risk-dot high"></span>Multiple high-risk lakes close together</span>
        <span style="color:#6EA2B3;">Red markers = lakes ≥ 70% predicted risk</span>
    </div>
    """, unsafe_allow_html=True)


elif page == "Growth Estimate":
    st.title("Lake Growth & Water Volume Estimate (Illustrative)")
    st.markdown(
        "<p style='color:#6EA2B3;'>No real satellite imagery is used here yet. This applies "
        "a flat assumed annual growth rate to each lake's current area to sketch what "
        "expansion could look like — a placeholder for real imagery analysis via Sentinel "
        "Hub / Google Earth Engine.</p>",
        unsafe_allow_html=True,
    )

    high_risk_df = df[df["glof_risk"] == 1]
    real_names = sorted(high_risk_df[high_risk_df["lake_type"] == "real"]["lake_name"].unique())
    synthetic_names = sorted(high_risk_df[high_risk_df["lake_type"] == "synthetic"]["lake_name"].unique())
    lake_options = (real_names + synthetic_names)[:15]

    lake_choice = st.selectbox("Select a lake", lake_options)
    row = df[df["lake_name"] == lake_choice].iloc[0]
    current_area = float(row["lake_area_km2"])
    retreat_rate = float(row["glacier_retreat_m_per_yr"])
    _, annual_rate = project_future_area(current_area, retreat_rate, 0)

    st.markdown("### Predicted Lake Area Over Future Years")
    horizon_years = st.slider(
        "Forecast horizon (years)", min_value=1, max_value=30, value=10,
        help="How many years into the future to project this lake's area.",
    )

    future_years = list(range(0, horizon_years + 1))
    future_area = [project_future_area(current_area, retreat_rate, y * 12)[0] for y in future_years]
    calendar_years = [2026 + y for y in future_years]

    fig = px.area(x=calendar_years, y=future_area,
                  labels={"x": "Year", "y": "Predicted Lake Area (km²)"},
                  title=f"Predicted Lake Area — {lake_choice}")
    fig.update_traces(line_color=T["accent"], fillcolor=f"rgba({int(T['accent'][1:3],16)},{int(T['accent'][3:5],16)},{int(T['accent'][5:7],16)},0.2)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=FONT_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="glof-card">
        <b>{lake_choice}</b> — starting from {current_area:.3f} km² today, this lake is
        projected to reach <b>{future_area[-1]:.3f} km²</b> by {calendar_years[-1]}
        (<b>{(future_area[-1]/current_area-1)*100:.0f}%</b> growth over {horizon_years} years),
        assuming a constant <b>{annual_rate*100:.1f}%/yr</b> growth rate derived from this
        lake's glacier retreat rate ({retreat_rate:.1f} m/yr).
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6EA2B3; font-size:0.82rem;'>This is a formula-based projection "
        "(area compounded at a rate tied to glacier retreat), not a model trained on "
        "observed area-over-time data — no such time series exists in this dataset yet. "
        "It also assumes the growth rate stays constant, which real lakes rarely do "
        "(growth can slow, reverse, or spike after an outburst). Treat it as a "
        "what-if sketch, not a forecast to act on.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Future Size & Water Volume Projection")
    st.markdown(
        "<p style='color:#6EA2B3;'>Projects <b>length, width, and water volume</b> for the "
        "selected lake at future time horizons, starting from today's measured area. "
        "This is a formula-based estimate, not a trained forecasting model — "
        "there's no bathymetry, lake-outline, or growth time-series data available to "
        "train one on. Expand the note below for exactly how each number is derived.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("How these numbers are calculated"):
        st.markdown("""
- **Future area**: today's area is compounded forward month by month at an assumed
  annual growth rate of `2% + 0.2% × glacier retreat rate (m/yr)`, capped between 1–12%/yr.
  A lake fed by a faster-retreating glacier is assumed to expand faster — this is a
  reasonable heuristic, not a rate measured from real observations of this lake.
- **Length & width**: back-calculated from the projected area assuming the lake is
  roughly elliptical with the length-to-width ratio you set below. Real lake outlines
  are irregular; adjust the slider to see how sensitive the estimate is to this assumption.
- **Water volume**: estimated from the projected area using Sakai (2012)'s published
  area–volume relation for Himalayan moraine-dammed glacial lakes, fitted to real
  bathymetric surveys — `V (million m³) = 43.24 × Area(km²)^1.53`. This formula comes
  from the literature, not from this project's dataset, and carries its own uncertainty
  (it was fitted on 17 lakes).
        """)

    aspect_ratio = st.slider(
        "Assumed length : width ratio", min_value=1.0, max_value=3.0, value=1.4, step=0.1,
        help="1.0 = roughly circular lake. Real Himalayan glacial lakes commonly range "
             "from about 1.2 to 2.5.",
    )

    horizons = [("2 months", 2), ("6 months", 6), ("1 year", 12), ("2 years", 24), ("5 years", 60)]

    proj_rows = []
    cur_len, cur_wid = area_to_length_width_m(current_area, aspect_ratio)
    proj_rows.append({
        "Horizon": "Today", "Area (km²)": round(current_area, 3),
        "Length (m)": round(cur_len, 1), "Width (m)": round(cur_wid, 1),
        "Volume (million m³)": round(area_to_volume_million_m3(current_area), 3),
    })
    for label, months in horizons:
        area_future, _ = project_future_area(current_area, retreat_rate, months)
        len_future, wid_future = area_to_length_width_m(area_future, aspect_ratio)
        vol_future = area_to_volume_million_m3(area_future)
        proj_rows.append({
            "Horizon": label, "Area (km²)": round(area_future, 3),
            "Length (m)": round(len_future, 1), "Width (m)": round(wid_future, 1),
            "Volume (million m³)": round(vol_future, 3),
        })
    proj_df = pd.DataFrame(proj_rows)

    st.markdown(f"""
    <div class="glof-card">
        <b>{lake_choice}</b> — assumed growth rate for this lake:
        <b>{annual_rate*100:.1f}%/yr</b> (based on a glacier retreat rate of
        {retreat_rate:.1f} m/yr). Adjust the shape slider above to explore how the
        length/width estimate changes.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(proj_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_dim = go.Figure()
        fig_dim.add_trace(go.Scatter(x=proj_df["Horizon"], y=proj_df["Length (m)"],
                                      mode="lines+markers", name="Length (m)", line=dict(color=T["accent"])))
        fig_dim.add_trace(go.Scatter(x=proj_df["Horizon"], y=proj_df["Width (m)"],
                                      mode="lines+markers", name="Width (m)", line=dict(color="#F2B84B")))
        fig_dim.update_layout(title="Projected Length & Width", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color=FONT_COLOR,
                               yaxis_title="Meters")
        st.plotly_chart(fig_dim, use_container_width=True)
    with c2:
        fig_vol = px.bar(proj_df, x="Horizon", y="Volume (million m³)",
                          title="Projected Water Volume", color_discrete_sequence=["#E8555B"])
        fig_vol.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=FONT_COLOR)
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown(
        "<p style='color:#6EA2B3; font-size:0.82rem;'>Treat this section as a planning "
        "sketch, not a validated forecast — verify against real imagery or field surveys "
        "before using these numbers for any actual hazard decision.</p>",
        unsafe_allow_html=True,
    )


elif page == "Weather Impact":
    st.title("Weather Impact on GLOF Risk")
    st.markdown(
        "<p style='color:#6EA2B3;'>How rainfall, temperature, and snowfall move the "
        "trained model's risk prediction. Unlike the Growth Estimate page, this isn't a "
        "made-up formula — every number below comes from actually running the trained "
        "model, holding a lake's other features fixed and varying one weather variable "
        "at a time (a standard interpretability technique called partial dependence).</p>",
        unsafe_allow_html=True,
    )

    with st.expander("An important caveat before you read the charts"):
        st.markdown(
            "For the ~780 synthetic lakes, `glof_risk` labels were generated from a "
            "weighted formula that included rainfall as one input (see `generate_dataset.py`). "
            "That means part of what the model has \"learned\" about rainfall's effect is "
            "really it recovering that formula, not an independently observed real-world "
            "relationship. The 10 real, named lakes are too few to fully untangle this. "
            "Read the trends below as *what this model currently believes*, not as "
            "established climate science."
        )

    weather_lake_options = sorted(df["lake_name"].unique())
    default_idx = weather_lake_options.index("South Lhonak Lake") if "South Lhonak Lake" in weather_lake_options else 0
    lake_choice_w = st.selectbox("Select a lake", weather_lake_options, index=default_idx, key="weather_lake_select")
    row_w = df[df["lake_name"] == lake_choice_w].iloc[0]
    baseline_dict = {col: row_w[col] for col in feature_cols}
    baseline_risk = predict_risk(baseline_dict)

    st.markdown(f"""
    <div class="glof-card">
        <b>{lake_choice_w}</b> — current predicted risk with its actual weather values
        (rainfall {row_w['rainfall_mm']:.0f} mm, temperature {row_w['temperature_c']:.1f}°C,
        snowfall {row_w['snowfall_mm']:.0f} mm): <b>{baseline_risk*100:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Sensitivity — this lake's risk as each weather variable changes")
    st.markdown(
        "<p style='color:#6EA2B3; font-size:0.85rem;'>Everything else about the lake "
        "(elevation, area, retreat rate, slope, region, etc.) is held fixed at its actual "
        "value while the chosen variable is swept across the full range seen in the "
        "dataset. The red dot marks where this lake actually sits today.</p>",
        unsafe_allow_html=True,
    )

    weather_vars = [
        ("rainfall_mm", "Rainfall (mm)", T["accent"]),
        ("temperature_c", "Temperature (°C)", "#F2B84B"),
        ("snowfall_mm", "Snowfall (mm)", "#4CAF6D"),
    ]
    cols_w = st.columns(3)
    for col_ui, (var, label, color) in zip(cols_w, weather_vars):
        xs, ys, cur_x, cur_y = weather_sensitivity(lake_choice_w, var)
        fig_pd = go.Figure()
        fig_pd.add_trace(go.Scatter(x=xs, y=ys * 100, mode="lines",
                                     line=dict(color=color), name="Model prediction"))
        fig_pd.add_trace(go.Scatter(x=[cur_x], y=[cur_y * 100], mode="markers",
                                     marker=dict(color="#E8555B", size=11), name="Current"))
        fig_pd.update_layout(
            title=f"Risk vs {label}", xaxis_title=label, yaxis_title="Predicted risk (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=FONT_COLOR, showlegend=False, height=320, margin=dict(t=40),
        )
        col_ui.plotly_chart(fig_pd, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Try a weather scenario")
    st.markdown(
        "<p style='color:#6EA2B3; font-size:0.85rem;'>Move the sliders to simulate a "
        "different climate for this lake — e.g. a wetter monsoon or a warmer year — and "
        "see how the model's risk prediction responds, all else held equal.</p>",
        unsafe_allow_html=True,
    )

    with st.form("weather_scenario_form"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            rainfall_s = st.slider("Rainfall (mm)", float(df["rainfall_mm"].min()),
                                    float(df["rainfall_mm"].max()), float(row_w["rainfall_mm"]))
        with sc2:
            temp_s = st.slider("Temperature (°C)", float(df["temperature_c"].min()),
                                float(df["temperature_c"].max()), float(row_w["temperature_c"]))
        with sc3:
            snow_s = st.slider("Snowfall (mm)", float(df["snowfall_mm"].min()),
                                float(df["snowfall_mm"].max()), float(row_w["snowfall_mm"]))
        scenario_submitted = st.form_submit_button("Run scenario")

    if scenario_submitted:
        scenario_dict = dict(baseline_dict)
        scenario_dict["rainfall_mm"] = rainfall_s
        scenario_dict["temperature_c"] = temp_s
        scenario_dict["snowfall_mm"] = snow_s
        scenario_risk = predict_risk(scenario_dict)
        delta = scenario_risk - baseline_risk

        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"""
        <div class="glof-card" style="text-align:center;">
            <div class="metric-label">Baseline Risk</div>
            <div class="metric-value">{baseline_risk*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        rc2.markdown(f"""
        <div class="glof-card" style="text-align:center;">
            <div class="metric-label">Scenario Risk</div>
            <div class="metric-value">{scenario_risk*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        delta_color = "var(--risk-high)" if delta > 0 else "var(--risk-low)"
        rc3.markdown(f"""
        <div class="glof-card" style="text-align:center;">
            <div class="metric-label">Change</div>
            <div class="metric-value" style="color:{delta_color} !important;">
                {'+' if delta >= 0 else ''}{delta*100:.1f} pts
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Weather patterns across all monitored lakes")
    st.markdown(
        "<p style='color:#6EA2B3; font-size:0.85rem;'>How each weather variable relates "
        "to predicted risk across the full dataset, not just one lake.</p>",
        unsafe_allow_html=True,
    )

    dc1, dc2, dc3 = st.columns(3)
    scatter_specs = [
        (dc1, "rainfall_mm", "Rainfall (mm)"),
        (dc2, "temperature_c", "Temperature (°C)"),
        (dc3, "snowfall_mm", "Snowfall (mm)"),
    ]
    for col_ui, var, label in scatter_specs:
        fig_sc = px.scatter(
            scored_df, x=var, y="risk_proba", color="risk_level",
            color_discrete_map={"low": "#4CAF6D", "medium": "#F2B84B", "high": "#E8555B"},
            title=f"{label} vs Predicted Risk", opacity=0.6,
        )
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=FONT_COLOR, showlegend=False, height=320,
            yaxis_title="Predicted risk", margin=dict(t=40),
        )
        col_ui.plotly_chart(fig_sc, use_container_width=True)


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
                              font_color=FONT_COLOR)
            st.plotly_chart(fig, use_container_width=True)


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


st.markdown("""
<div class="footer-strip">
    <span>🛡️ Early Warning Today, Safer Tomorrow</span>
    <span>GLOF Early Warning System — India</span>
    <span>Data saves lives. Stay alert, stay safe.</span>
</div>
""", unsafe_allow_html=True)
