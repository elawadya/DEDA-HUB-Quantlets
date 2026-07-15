<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: Credit Risk Prediction in P2P Lending

Description: Empirical analysis of credit-risk prediction in peer-to-peer lending using the Bondora public loan dataset. The project constructs alternative default labels, compares logistic regression, LightGBM, TabPFN-3, and Bondora's published Probability of Default score, evaluates discrimination and calibration across prediction horizons, and applies conformal prediction to quantify uncertainty under country, time, and label shifts. Visualizations cover country-level default patterns, label sensitivity, probability calibration, Brier-score decomposition, vintage effects, conformal coverage, online adaptation, label robustness, and FDR-controlled approval.

Keywords: peer-to-peer lending, credit risk, Bondora, default prediction, probability of default, calibration, AUC, Brier score, conformal prediction, split conformal prediction, online conformal prediction, distribution shift, label robustness, false discovery rate, country-specific evaluation, prediction horizon, logistic regression, LightGBM, TabPFN, statistical learning, financial econometrics, data analysis, visualization

Author: Wolfgang Karl Härdle, Jinzhao Fu

Submitted_By: Jinzhao Fu

Submitted_To: DEDA-Seminar_Courselet

Institution: Humboldt University of Berlin

Email: jinzhao.fu@student.hu-berlin.de

Created_On: 2026-06-22

Code_Files: code/bondora_credit_risk_analysis.py, code/conformal_experiments.py, code/run_tabpfn_remote.py

Data_Files: LoanData.csv

Data_Source_URL: https://www.kaggle.com/datasets/marcobeyer/bondora-p2p-loans/data

Output_Files: data/closed_loans.csv, data/loans_clean.csv, data/step5_predictions.csv, data/conformal_master.csv, figures/country_default_rates.png, figures/default_label_sensitivity.png, figures/pod_calibration_1year.png, figures/pod_calibration_lifetime.png, figures/brier_score_decomposition.png, figures/entry_vintage_default_rates.png, figures/phase_a_summary.csv, figures/split_conformal_coverage_width.png, figures/group_conditional_coverage.png, figures/distribution_shift_diagnostic.png, figures/online_adaptation_coverage.png, figures/label_robustness_coverage.png, figures/conformal_selection_fdr.png, figures/tabpfn_conformal_coverage.png

Libraries: numpy, pandas, matplotlib, scikit-learn, lightgbm, pyarrow, tabpfn-client

Programming_Language: Python

Quantlet_Class: Application Quantlet

Quantlet_Type: Analysis, Visualization, Financial Econometrics, Machine Learning

Version: 1.0

```
