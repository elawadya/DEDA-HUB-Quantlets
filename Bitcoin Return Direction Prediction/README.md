Name of Quantlet: Bitcoin Return Direction Prediction

Description: >
  BTC-USD daily return direction predicts (Up/Down) using Random Forest,
  XGBoost, LightGBM, Voting Ensemble, and LSTM with Fear & Greed and VADER
  sentiment features. Includes McNemar test and trading strategy evaluation 
  with transaction costs.

Keywords: >
  bitcoin, return direction prediction, random forest, 
  xgboost, lightgbm, lstm, voting ensemble, fear and greed,
  vader sentiment, market regime, trading strategy, 
  efficient market hypothesis, 

Author: Wolfgang Karl Härdle, Albraa Elawady

Submitted_By: Albraa Elawady

Submitted_To: DEDA-Seminar_Courselet SS2026

Institution: Humboldt-Universität zu Berlin

Created_On: 2026-07-19

Code_Files: >
  Bitcoin_return_direction_prediction.ipynb

Data_Files: >
  btc_raw.csv, feargreed_raw.csv, bitcoin_sentiments_21_24.csv, vader_daily.csv

Output_Files: >
  model_results.csv, model_accuracy.png, roc_curves.png,
  shap_beeswarm.png, shap_bar.png, bootstrap_ci.png,
  mcnemar_heatmap_tech.png, mcnemar_heatmap_fg.png,
  confusion_matrix_best.png, feature_importance_main.png,
  feature_importance_overlap.png, market_regime.png,
  trading_strategy.png, best_params.json, best_params_ol.json,
  lstm_best_params.json

Libraries: >
  numpy, pandas, yfinance, matplotlib, sklearn, scipy,
  tensorflow, keras, optuna, xgboost, lightgbm, shap

Programming_Language: Python

Quantlet_Class: Application Quantlet

Quantlet_Type: Analysis, Predictive Modeling, Financial Econometrics

Version: 1.0
