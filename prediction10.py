import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Spanish GP qualifying session (Round 8)
session = fastf1.get_session(2024, 8, "Q")
session.load()

# Extract lap data
laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Team"]].dropna()

# Convert time to seconds
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

# Clean air race pace (adjusted manually or from historical data)
clean_air_race_pace = {
    "VER": 93.2, "HAM": 94.0, "LEC": 93.5, "NOR": 93.6, "ALO": 94.7,
    "PIA": 93.3, "RUS": 93.8, "SAI": 94.4, "STR": 95.2, "HUL": 95.4,
    "OCO": 95.6, "ALB": 95.3, "GAS": 95.5
}

# Team performance scores
team_points = {
    "Red Bull": 105, "Mercedes": 141, "McLaren": 246, "Ferrari": 94,
    "Aston Martin": 14, "Alpine": 7, "Williams": 37,
    "Haas": 20, "Kick Sauber": 6, "Racing Bulls": 8
}
max_pts = max(team_points.values())
team_score = {team: pts / max_pts for team, pts in team_points.items()}

# Weather (default dry)
rain_probability = 0.0
temperature = 20.0

# Build feature set from fastest lap per driver
driver_info = []
for drv in laps["Driver"].unique():
    try:
        fastest_lap = laps[laps["Driver"] == drv].sort_values("LapTime").iloc[0]
        if pd.isna(fastest_lap["LapTime"]):
            continue
        driver_name = session.get_driver(drv).get("full_name", drv)
        team = fastest_lap["Team"]
        lap_time = fastest_lap["LapTime (s)"]
        
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
        print(f"Skipping driver {drv}: {e}")

# Build DataFrame
qualifying_df = pd.DataFrame(driver_info)

# Merge sector times
merged = qualifying_df.merge(sector_times[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")

# Feature matrix and labels
X = merged[[
    "QualifyingTime (s)", "RainProbability", "Temperature",
    "TeamPerformanceScore", "CleanAirRacePace (s)"
]]
y = laps.groupby("Driver")["LapTime (s)"].mean().reindex(merged["Driver"])

# Handle missing
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
model.fit(X_train, y_train)

# Predictions
merged["PredictedRaceTime (s)"] = model.predict(X_imputed)
results = merged.sort_values("PredictedRaceTime (s)").reset_index(drop=True)

# Output results
print("\n🏁 Predicted 2025 Spanish Grand Prix Results 🏁")
print(results[["Driver", "DriverName", "Team", "PredictedRaceTime (s)"]])
print(f"\n📉 Model MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} seconds")

# Clean Air Pace vs Prediction
plt.figure(figsize=(10, 6))
plt.scatter(results["CleanAirRacePace (s)"], results["PredictedRaceTime (s)"])
for i, row in results.iterrows():
    plt.annotate(row["Driver"], (row["CleanAirRacePace (s)"], row["PredictedRaceTime (s)"]),
                 xytext=(4, 4), textcoords="offset points")
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Prediction vs Clean Air Pace - Spanish GP")
plt.tight_layout()
plt.show()

# Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_, color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance in Race Prediction")
plt.tight_layout()
plt.show()

# Podium
print("\n🏆 Predicted Podium - Spanish GP 🏆")
for i in range(3):
    row = results.iloc[i]
    print(f"🥇 P{i+1}: {row['Driver']} - {row['DriverName']} ({row['Team']})")
