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

# Load the 2024 Emilia Romagna GP qualifying session
session_2024 = fastf1.get_session(2024, 7, "Q")
session_2024.load()

# Extract lap data
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna()

# Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()
sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# Clean air race pace (approximate)
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}

# Qualifying data (2025 simulated)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [
        74.704, 74.962, 74.670, 74.807, 75.432,
        75.473, 75.604, 76.613, 75.765, 75.581,
        75.787, 75.431, 76.518
    ]
})

# Add clean air race pace
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Simulated static weather values
rain_probability = 0.0
temperature = 20.0

# Final qualifying time (unaffected by rain here)
qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructor points (relative team strength)
team_points = {
    "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Driver-team mapping
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin", "ALB": "Williams"
}

# Add team + performance score
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Merge with sector times
merged_data = qualifying_2025.merge(
    sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left"
)
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "CleanAirRacePace (s)"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=34)

# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=34)
model.fit(X_train, y_train)

# Make predictions
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)

# Display results
print("\n🏁 Predicted 2025 Emilia Romagna GP Results 🏁")
print(final_results[["Driver", "Team", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot race pace vs prediction
plt.figure(figsize=(10, 6))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, row in final_results.iterrows():
    plt.annotate(row["Driver"], (row["CleanAirRacePace (s)"], row["PredictedRaceTime (s)"]),
                 xytext=(5, 5), textcoords="offset points")
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Clean Air Pace vs Predicted Race Time")
plt.tight_layout()
plt.show()

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_, color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance in Prediction")
plt.tight_layout()
plt.show()

# Display podium
print("\n🏆 Predicted Podium 🏆")
for i in range(3):
    row = final_results.iloc[i]
    print(f"🥇 P{i+1}: {row['Driver']} ({row['Team']}) — {row['PredictedRaceTime (s)']:.2f} sec")
