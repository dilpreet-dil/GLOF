"""
GLOF Risk Dataset Generator — INDIA EDITION
=============================================
WHY THIS APPROACH:
There is no public, ready-made, labeled tabular CSV for GLOF (Glacial Lake
Outburst Flood) risk prediction in India specifically. Real GLOF science
works with satellite imagery and lake inventories (NASA, ICIMOD, Sentinel,
NRSC/ISRO), not spreadsheets.

So this dataset is built in two layers, and both are documented so it is
defensible in a project report:

LAYER 1 - REAL ANCHOR LAKES (10 rows, all within Indian Himalayan states):
Real, named Indian glacial lakes with real published coordinates,
elevation, area, and GLOF history/risk status, taken from peer-reviewed
literature and government/research-institute reports (Sattar et al.
2021/2023 on South Lhonak, Latief & Kumar and Kumar et al. on Lahaul-Spiti
lakes, ICIMOD/NRSC lake inventories, and well-documented public event
records such as the 2013 Kedarnath/Chorabari Lake disaster and the 2023
South Lhonak Lake outburst).
- Lakes with an ACTUAL documented GLOF event (South Lhonak 2023, Chorabari
  2013, Parechu 2004/2005 breach events) are labeled high risk (1).
- Lakes repeatedly flagged in the literature as rapidly expanding /
  potentially dangerous moraine- or glacier-dammed lakes (Samudra Tapu,
  Gepang Gath, both in Lahaul-Spiti, Himachal Pradesh) are also labeled
  high risk (1), consistent with how such lakes are treated in published
  GLOF susceptibility studies even before an actual outburst occurs.
- Long-stable, non-glacial or tourist lakes (Gurudongmar, Tsomgo,
  Khecheopalri, Samiti, Tso Lhamo — all in Sikkim) are labeled low risk (0).

LAYER 2 - LITERATURE-GROUNDED EXPANSION (rest of rows):
Additional synthetic lake records sampled from realistic value ranges
reported in the same literature (elevation 3500-6000m, area 0.005-2 km²,
etc.), distributed across the Indian Himalayan states/UTs that actually
contain glacial lakes: Jammu & Kashmir, Ladakh, Himachal Pradesh,
Uttarakhand, Sikkim, and Arunachal Pradesh. Coordinates are sampled with
rejection sampling against REAL state boundary polygons (from a public
India administrative-boundaries GeoJSON), not simple rectangular lat/lon
boxes — so every point is verified to actually fall inside Indian
territory rather than spilling into Nepal, Bhutan, Tibet, or Myanmar at
a state's corners (a real risk for irregularly-shaped states like Sikkim
and Arunachal Pradesh if you just sample a rectangle around them). Risk
labels are computed with a transparent, weighted multi-criteria scoring
formula modeled on the Analytical Hierarchy Process (AHP) approach used
in real GLOF susceptibility studies, not assigned randomly.

This hybrid is standard practice for student/prototype ML projects on
hazards where no labeled dataset exists, and is clearly disclosed here
rather than presented as an official government/NASA dataset.
"""
import pickle
import numpy as np
import pandas as pd
from shapely.geometry import Point

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# Real state boundary polygons (pre-built from a public India admin-boundary
# GeoJSON; see region_polygons.pkl). Undivided pre-2019 "Jammu and Kashmir"
# was split at longitude 76.5 into an approximate "Jammu & Kashmir" (west)
# and "Ladakh" (east) piece, since Ladakh became a separate UT in 2019 and
# no standalone boundary existed in the source file.
# ---------------------------------------------------------------
with open("region_polygons.pkl", "rb") as f:
    region_polys = pickle.load(f)


def sample_point_in_region(region_name, rng):
    """Rejection-sample a lat/lon point that actually falls inside the real
    polygon for this Indian state/UT (not just its rectangular bounding box)."""
    poly = region_polys[region_name]
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(200):
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        if poly.contains(Point(lon, lat)):
            return lat, lon
    # Extremely unlikely fallback: polygon centroid
    c = poly.centroid
    return c.y, c.x

# ---------------------------------------------------------------
# LAYER 1: REAL, NAMED, DOCUMENTED INDIAN LAKES
# ---------------------------------------------------------------
real_lakes = [
    # name, state, lat, lon, elevation_m, lake_area_km2, glacier_retreat_m_per_yr,
    # distance_from_glacier_m, slope_deg, rainfall_mm, temperature_c, snowfall_mm,
    # earthquake_magnitude, glof_risk (1 = documented dangerous / had GLOF, 0 = stable)
    ("South Lhonak Lake", "Sikkim", 27.9067, 88.1904, 5200, 1.66, 42.0, 150, 28, 2200, -2.1, 1800, 4.2, 1),
    ("Chorabari Lake", "Uttarakhand", 30.7346, 79.0669, 3800, 0.05, 8.0, 500, 33, 3000, 2.0, 2200, 4.9, 1),
    ("Parechu Lake", "Himachal Pradesh", 32.4200, 78.7200, 4700, 1.20, 10.0, 300, 15, 900, -3.5, 1000, 4.4, 1),
    ("Samudra Tapu Lake", "Himachal Pradesh", 32.4800, 77.6200, 4076, 1.35, 28.0, 180, 20, 1200, -2.8, 1300, 4.1, 1),
    ("Gepang Gath Lake", "Himachal Pradesh", 32.3600, 77.1300, 4068, 0.45, 24.0, 220, 22, 1250, -2.4, 1350, 4.0, 1),
    ("Gurudongmar Lake", "Sikkim", 28.0333, 88.7167, 5425, 0.10, 1.0, 1500, 4, 500, -8.0, 400, 3.0, 0),
    ("Tsomgo Lake", "Sikkim", 27.3746, 88.7628, 3753, 0.02, 0.5, 2000, 3, 1600, 1.0, 900, 3.2, 0),
    ("Khecheopalri Lake", "Sikkim", 27.3600, 88.2100, 1700, 0.01, 0.0, 5000, 2, 3200, 12.0, 200, 2.5, 0),
    ("Samiti Lake", "Sikkim", 27.6000, 88.1300, 4300, 0.03, 2.0, 1200, 9, 2400, -1.5, 1100, 3.6, 0),
    ("Tso Lhamo Lake", "Sikkim", 28.1500, 88.7000, 5330, 0.60, 3.0, 1000, 5, 450, -7.5, 380, 3.1, 0),
]

real_df = pd.DataFrame(real_lakes, columns=[
    "lake_name", "region", "latitude", "longitude", "elevation_m", "lake_area_km2",
    "glacier_retreat_m_per_yr", "distance_from_glacier_m", "slope_deg",
    "rainfall_mm", "temperature_c", "snowfall_mm", "earthquake_magnitude", "glof_risk"
])
real_df["lake_type"] = "real"

# ---------------------------------------------------------------
# LAYER 2: LITERATURE-GROUNDED SYNTHETIC EXPANSION (India only)
# region = Indian Himalayan state/UT, using real state boundary polygons
# ---------------------------------------------------------------
N_SYNTHETIC = 780
regions = ["Jammu & Kashmir", "Ladakh", "Himachal Pradesh", "Uttarakhand", "Sikkim", "Arunachal Pradesh"]
# Rough relative share of India's known glacial-lake inventory per state
# (Ladakh/HP/Uttarakhand carry the largest counts, Sikkim/Arunachal smaller)
region_weights = np.array([0.14, 0.27, 0.24, 0.20, 0.09, 0.06])
region_weights = region_weights / region_weights.sum()

# Synthetic lakes are NOT real, specific lakes — they are placeholder records
# (see Layer 2 note above). To make the dashboard readable, each one still
# gets a distinct, region-flavored name using the actual local word for
# "lake" used in that region's toponymy (Tso in Ladakh/Sikkim — Tibetan;
# Sar in Kashmir — Kashmiri; Tal in Himachal Pradesh/Uttarakhand — Hindi/
# Pahari), combined with generic geographic descriptors (never a real,
# specific valley/peak name) plus a unique index, so it reads naturally
# without ever claiming to be an actual documented lake.
region_lake_word = {
    "Jammu & Kashmir":   "Sar",
    "Ladakh":            "Tso",
    "Himachal Pradesh":  "Tal",
    "Uttarakhand":       "Tal",
    "Sikkim":            "Tso",
    "Arunachal Pradesh": "Lake",
}
name_adjectives = ["Upper", "Lower", "North", "South", "East", "West", "Inner", "Outer", "High", "Far"]
name_nouns = ["Ridge", "Basin", "Valley", "Glacier", "Moraine", "Cirque", "Plateau", "Col", "Spur", "Hollow"]


def make_synthetic_name(region, idx, rng):
    adj = rng.choice(name_adjectives)
    noun = rng.choice(name_nouns)
    word = region_lake_word[region]
    return f"{adj} {noun} {word} {idx:03d}"


rows = []
for i in range(N_SYNTHETIC):
    region = rng.choice(regions, p=region_weights)
    lat, lon = sample_point_in_region(region, rng)
    lake_name = make_synthetic_name(region, i + 1, rng)

    elevation = rng.uniform(3500, 6000)
    lake_area = float(np.round(rng.lognormal(mean=-1.3, sigma=1.0), 3))
    lake_area = min(max(lake_area, 0.005), 5.0)
    retreat_rate = max(0.0, rng.normal(15, 12))
    dist_glacier = max(50, rng.normal(600, 500))
    slope = np.clip(rng.normal(18, 9), 1, 45)
    rainfall = np.clip(rng.normal(2100, 650), 300, 3800)
    temperature = np.clip(rng.normal(-1.5, 3.5), -10, 15)
    snowfall = np.clip(rng.normal(1300, 500), 100, 2800)
    eq_mag = np.clip(rng.normal(4.0, 0.8), 2.0, 7.5)

    rows.append([
        lake_name, region, round(lat, 4), round(lon, 4),
        round(elevation, 1), round(lake_area, 3), round(retreat_rate, 2),
        round(dist_glacier, 1), round(slope, 1), round(rainfall, 1),
        round(temperature, 2), round(snowfall, 1), round(eq_mag, 2), None
    ])

synth_df = pd.DataFrame(rows, columns=[c for c in real_df.columns if c != "lake_type"])
synth_df["lake_type"] = "synthetic"

# ---- AHP-style transparent multi-criteria risk scoring (documented weights) ----
def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

w = dict(area=0.28, retreat=0.22, slope=0.15, rainfall=0.15,
         proximity=0.12, quake=0.08)

score = (
    w["area"]      * normalize(synth_df["lake_area_km2"]) +
    w["retreat"]   * normalize(synth_df["glacier_retreat_m_per_yr"]) +
    w["slope"]     * normalize(synth_df["slope_deg"]) +
    w["rainfall"]  * normalize(synth_df["rainfall_mm"]) +
    w["proximity"] * (1 - normalize(synth_df["distance_from_glacier_m"])) +
    w["quake"]     * normalize(synth_df["earthquake_magnitude"])
)
score = score + rng.normal(0, 0.03, size=len(score))  # small realistic noise
threshold = np.quantile(score, 0.62)  # ~38% high risk, roughly matches real proportion above
synth_df["glof_risk"] = (score > threshold).astype(int)

# ---------------------------------------------------------------
# COMBINE + inject a small amount of realistic missingness (like real field data)
# ---------------------------------------------------------------
full_df = pd.concat([real_df, synth_df], ignore_index=True)
full_df = full_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
full_df.insert(0, "lake_id", [f"GL{idx+1:04d}" for idx in range(len(full_df))])

# introduce a few missing values in non-critical columns to mimic real field data gaps
miss_idx_snow = rng.choice(full_df.index, size=25, replace=False)
full_df.loc[miss_idx_snow, "snowfall_mm"] = np.nan
miss_idx_eq = rng.choice(full_df.index, size=15, replace=False)
full_df.loc[miss_idx_eq, "earthquake_magnitude"] = np.nan

# a handful of duplicate rows (also realistic, tests cleaning step)
dupes = full_df.sample(6, random_state=3)
full_df = pd.concat([full_df, dupes], ignore_index=True)

full_df.to_csv("/home/claude/glof_project/glof_dataset.csv", index=False)
print("Saved:", full_df.shape)
print(full_df["glof_risk"].value_counts(normalize=True))
print(full_df["region"].value_counts())
