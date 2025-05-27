## Developed by Ankita Patra

# 2025_f1_predictions

# 🏎️ F1 Predictions 2025 – Machine Learning Race Simulator

Welcome to the **F1 Predictions 2025** project! This repository applies **machine learning, FastF1 telemetry, and historical race data** to predict the finishing order of each Formula 1 Grand Prix in the 2025 season — before the race even starts.

We currently include predictions for:
- 🇮🇹 **Emilia Romagna Grand Prix** (Imola)
- 🇪🇸 **Spanish Grand Prix** (Barcelona)

## 🚀 Project Overview

This project uses a **Gradient Boosting Regressor** to predict driver race times based on:

- FastF1 API data (sector times, lap performance)
- 2024 race and qualifying results
- 2025 qualifying session inputs
- Driver race pace performance (in clean air)
- Team performance using constructor standings
- Static or optional real weather features

Each race is predicted independently using a structured data pipeline and statistical modeling.


## 📊 Data Sources

- 🟢 **FastF1 API**: To retrieve lap-by-lap and sector-by-sector telemetry
- 🟢 **2024 Qualifying Sessions**: Used as historical references
- 🟢 **2025 Simulated Qualifying Times**: Manually curated inputs
- 🟢 **Team Constructor Points**: Used to estimate car performance
- 🟡 *(Optional)* Weather Data via OpenWeatherMap API

*(Optional)* : if you want we can add these feature : 

    📌 Planned Improvements
---------------------------------------------------------

🌦️ Add live weather forecast support
🛞 Include pit stop strategy & tire compound data
🧪 Add uncertainty simulations using Monte Carlo
🌐 Deploy via Streamlit web app for public use
🏁 Train on full 2024 race calendar for improved generalization

## 🏁 Supported Races

### 🇮🇹 Emilia Romagna GP
- 📍 Track: Autodromo Enzo e Dino Ferrari (Imola)
- 📆 Date: Sunday, May 18, 2025
- 🕒 Session: 3:00 PM local time
- 📂 File: `prediction7.py`

### 🇪🇸 Spanish GP
- 📍 Track: Circuit de Barcelona-Catalunya
- 📆 Date: Sunday, June 1, 2025
- 🕘 Session: 9:00 AM local time
- 📂 File: `prediction9.py`

## 🧠 How It Works

1. **Data Collection**  
   - Automatically loads the qualifying session via FastF1
   - Extracts each driver's fastest lap, sector times, and team

2. **Preprocessing & Feature Engineering**  
   - Converts time values to seconds
   - Adds team score (based on constructor points)
   - Adds clean air race pace and weather inputs

3. **Model Training**  
   - Trains a **Gradient Boosting Regressor** to learn from the data

4. **Prediction**  
   - Estimates the expected race time for each driver
   - Sorts by predicted time to determine finishing order

5. **Evaluation**  
   - Prints the podium and computes **MAE (Mean Absolute Error)**

6. **Visualization**  
   - Generates scatter and bar plots to show model behavior and importance of each feature

---

## 🖥️ Dependencies

Make sure the following libraries are installed:

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib

# 📁 File Structure

2025_f1_predictions/
├── prediction1.py         # Australia GP
├── prediction2.py         # China GP
├── prediction7.py         # Emilia Romagna GP
├── prediction9.py         # Spanish GP
├── f1_cache/              # FastF1 telemetry cache
└── README.md              # This file

# 🔧 Usage
 run : python3 prediction9.py

# Output : 

(base) ankitapatra@Mac 2025_f1_predictions % python prediction10.py

core           INFO     Loading data for Monaco Grand Prix - Qualifying [v3.5.3]
req            INFO     Using cached data for session_info
req            INFO     Using cached data for driver_info
req            INFO     Using cached data for session_status_data
req            INFO     Using cached data for track_status_data
req            INFO     Using cached data for _extended_timing_data
req            INFO     Using cached data for timing_app_data
core           INFO     Processing timing data...
core        WARNING     Driver  3: Lap timing integrity check failed for 1 lap(s)
req            INFO     Using cached data for car_data
req            INFO     Using cached data for position_data
req            INFO     Using cached data for weather_data
req            INFO     Using cached data for race_control_messages
core           INFO     Finished loading data for 20 drivers: ['16', '81', '55', '4', '63', '1', '44', '22', '23', '10', '31', '3', '18', '27', '14', '2', '20', '11', '77', '24']

🏁 Predicted 2025 Spanish Grand Prix Results 🏁
   Driver DriverName             Team  PredictedRaceTime (s)
0     HUL        HUL     Haas F1 Team              84.784262
1     ALO        ALO     Aston Martin              85.448639
2     GAS        GAS           Alpine              85.581107
3     NOR        NOR          McLaren              86.028789
4     PIA        PIA          McLaren              86.042646
5     TSU        TSU               RB              86.232953
6     PER        PER  Red Bull Racing              86.287049
7     SAR        SAR         Williams              86.287049
8     VER        VER  Red Bull Racing              86.600324
9     ALB        ALB         Williams              86.914686
10    HAM        HAM         Mercedes              87.072801
11    LEC        LEC          Ferrari              87.091639
12    RIC        RIC               RB              87.149822
13    OCO        OCO           Alpine              87.391921
14    STR        STR     Aston Martin              87.542434
15    MAG        MAG     Haas F1 Team              87.996037
16    SAI        SAI          Ferrari              88.397413
17    ZHO        ZHO      Kick Sauber              89.113451
18    RUS        RUS         Mercedes              89.217104
19    BOT        BOT      Kick Sauber              91.517456

📉 Model MAE: 1.34 seconds
2025-05-26 21:39:32.088 python[13739:732025] The class 'NSSavePanel' overrides the method identifier.  This method is implemented by class 'NSWindow'

🏆 Predicted Podium - Spanish GP 🏆
🥇 P1: HUL - HUL (Haas F1 Team)
🥇 P2: ALO - ALO (Aston Martin)
🥇 P3: GAS - GAS (Alpine)


# 📈 Model Performance

    The model's accuracy is evaluated using Mean Absolute Error (MAE). A lower MAE means more precise timing predictions. MAE typically ranges between ~27 to 30 seconds, depending on race complexity and weather assumptions.

