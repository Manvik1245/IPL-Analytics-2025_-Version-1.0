IPL 2025 Analytics Engine
Version: 1.1 (ML Enhanced)

Language: Python 3.x

A powerful, command-line analytics engine designed to process IPL 2025 match data. This tool moves beyond basic scorecards to calculate advanced metrics like Impact Scores and Effective Scoring Indices, and now features a Machine Learning Module to predict future top performers using Random Forest regression.

🚀 Features
The engine processes data from three core datasets (batsman, bowler, matches) to generate insights across four categories:

1. Batting Analytics
True Impact Score: A weighted metric combining Average and Strike Rate relative to balls faced.

Effective Scoring Index (ESI): Measures scoring efficiency by factoring in "Balls Per Dismissal" (BPD).

Clutch Performance: Calculates the percentage of a team's runs scored by a player specifically in winning matches.

Dismissal Analysis: Breaks down how a player gets out (Caught vs. Bowled vs. LBW).

Conversion Rates: Tracks how often a player converts starts into 50+ scores.

2. Bowling Analytics
Bowling Impact Score: A formula rewarding wickets and dot balls while penalizing runs conceded.

Pressure Metrics: Tracks Dot Balls and Economy Rates.

Wicket Hauls: Identifies bowlers with the most 3+ wicket hauls.

Win Contribution: Calculates a bowler's wicket share in winning causes.

3. All-Rounder Analysis
MVP Leaderboard: Ranks players based on total "Impact Points" (Runs + Wickets * 25).

Role Classifier: Automatically categorizes players as Batting All-Rounder, Bowling All-Rounder, or Genuine All-Rounder based on performance ratios.

4. 🤖 Machine Learning Predictions (New!)
Run Scorer Predictor: Uses RandomForestRegressor to predict the highest run-getters based on batting position, boundaries, and running between wickets.

Wicket Taker Predictor: Predicts future wicket hauls based on economy rate, dot balls, and boundary suppression metrics.

Installation & Usage
1. Prerequisites
You need Python 3.7+ installed.

2. Install Dependencies
This project now relies on scikit-learn for its prediction models. Run the following command:
pip install pandas numpy matplotlib scikit-learn

Contributions are welcome! Please fork the repository and submit a pull request for any new metrics or bug fixes.
