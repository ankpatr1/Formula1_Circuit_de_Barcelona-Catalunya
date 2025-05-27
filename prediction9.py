import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Qualifying session for race 7 (Emilia Romagna)
session = fastf1.get_session(2024, 7, "Q")
session.load()

# Extract relevant laps
laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Team"]].dropna()

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps[f"{col} (s)"] = laps[col].dt.total_seconds()

# Aggregate sector times
sector_times = laps.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()
sector_times["TotalSectorTime (s)"] = (
    sector_times["Sector1Time (s)"] +
    sector_times["Sector2Time (s)"] +
    sector_times["Sector3Time (s)"]
)

# Clean air race pace data
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128, "ALB": 95.3, "GAS": 95.5
}

# Constructor points
team_points = {
    "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
}
max_pts = max(team_points.values())
team_score = {team: pts / max_pts for team, pts in team_points.items()}

# Weather fallback
rain_probability = 0.0
temperature = 20.0

# Extract driver info using driver abbreviations
driver_info = []

for drv in laps["Driver"].unique():
    try:
        fastest_lap = laps[laps["Driver"] == drv].sort_values("LapTime").iloc[0]
        if pd.isna(fastest_lap["LapTime"]):
            continue

        driver_data = session.get_driver(drv)
        driver_name = driver_data.get("full_name", drv)
        lap_time = fastest_lap["LapTime (s)"]
        team = fastest_lap["Team"]

        driver_info.append({
            "Driver": drv,
            "DriverName": driver_name,
            "Team": team,
            "QualifyingTime (s)": lap_time,
            "CleanAirRacePace (s)": clean_air_race_pace.get(drv, np.nan),
            "TeamPerformanceScore": team_score.get(team, np.nan),
            "RainProbability": rain_probability,
            "Temperature": temperature
        })
    except Exception as e:
        print(f"Skipping driver {drv} due to error: {e}")

# Build DataFrame
qualifying_df = pd.DataFrame(driver_info)

# Merge with sector data
merged_data = qualifying_df.merge(
    sector_times[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left"
)

# Prepare features
X = merged_data[[
    "QualifyingTime (s)", "RainProbability", "Temperature",
    "TeamPerformanceScore", "CleanAirRacePace (s)"
]]
y = laps.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
model.fit(X_train, y_train)

# Predict
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)
results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)

# Output results
print("\n🏁 Predicted 2025 Emilia Romagna GP Results 🏁")
print(results[["Driver", "DriverName", "Team", "PredictedRaceTime (s)"]])

# MAE
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Clean air pace vs prediction
plt.figure(figsize=(10, 6))
plt.scatter(results["CleanAirRacePace (s)"], results["PredictedRaceTime (s)"])
for i, row in results.iterrows():
    plt.annotate(row["Driver"], (row["CleanAirRacePace (s)"], row["PredictedRaceTime (s)"]),
                 xytext=(5, 5), textcoords="offset points")
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Clean Air Race Pace on Prediction")
plt.tight_layout()
plt.show()

# Feature importances
plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_, color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance in Prediction")
plt.tight_layout()
plt.show()

# Podium
podium = results.head(3)
print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']} - {podium.iloc[0]['DriverName']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']} - {podium.iloc[1]['DriverName']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']} - {podium.iloc[2]['DriverName']}")
