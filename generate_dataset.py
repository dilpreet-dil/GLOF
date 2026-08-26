
import pickle
import numpy as np
import pandas as pd
from shapely.geometry import Point

rng = np.random.default_rng(42)


with open("region_polygons.pkl", "rb") as f:
    region_polys = pickle.load(f)


def sample_point_in_region(region_name, rng):
    poly = region_polys[region_name]
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(200):
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        if poly.contains(Point(lon, lat)):
            return lat, lon
    
    c = poly.centroid
    return c.y, c.x


real_lakes = [
    
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


N_SYNTHETIC = 780
regions = ["Jammu & Kashmir", "Ladakh", "Himachal Pradesh", "Uttarakhand", "Sikkim", "Arunachal Pradesh"]

region_weights = np.array([0.14, 0.27, 0.24, 0.20, 0.09, 0.06])
region_weights = region_weights / region_weights.sum()


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
score = score + rng.normal(0, 0.03, size=len(score))  
threshold = np.quantile(score, 0.62)  
synth_df["glof_risk"] = (score > threshold).astype(int)


full_df = pd.concat([real_df, synth_df], ignore_index=True)
full_df = full_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
full_df.insert(0, "lake_id", [f"GL{idx+1:04d}" for idx in range(len(full_df))])


miss_idx_snow = rng.choice(full_df.index, size=25, replace=False)
full_df.loc[miss_idx_snow, "snowfall_mm"] = np.nan
miss_idx_eq = rng.choice(full_df.index, size=15, replace=False)
full_df.loc[miss_idx_eq, "earthquake_magnitude"] = np.nan


dupes = full_df.sample(6, random_state=3)
full_df = pd.concat([full_df, dupes], ignore_index=True)

full_df.to_csv("glof_dataset.csv", index=False)
print("Saved:", full_df.shape)
print(full_df["glof_risk"].value_counts(normalize=True))
print(full_df["region"].value_counts())
