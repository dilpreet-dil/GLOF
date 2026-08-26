# GLOF Risk Prediction — India

A machine learning project that predicts Glacial Lake Outburst Flood (GLOF) risk for glacial lakes across the Indian Himalaya (Jammu & Kashmir, Ladakh, Himachal Pradesh, Uttarakhand, Sikkim, Arunachal Pradesh). It uses lake and environmental characteristics such as elevation, lake area, glacier retreat rate, slope, rainfall, and seismic activity to classify a lake as high or low risk, and presents the results through an interactive dashboard.

This project was built to apply the complete machine learning workflow to a real-world environmental problem. It covers data generation, preprocessing, model training, evaluation, explainability, and deployment in a single application. While the project demonstrates the overall workflow, it should be viewed as a proof of concept rather than a production-ready early-warning system.

## What this is not

There are a few important limitations to be aware of. This project does not use real satellite imagery or remote sensing data. The "Lake Growth Estimate" page in the dashboard is a placeholder that applies a flat assumed growth rate to a lake's current area — it is not derived from analyzing actual images. There is also no live monitoring feed; the "Alert System" page shows example alerts generated from the trained model's current predictions on the existing dataset, not from any real-time source.

The project is intended as a foundation for that kind of system rather than a complete implementation. Adding satellite imagery and real-time monitoring is planned as future work.

## Data

The dataset combines two sources:

- **10 real lakes**, each entered manually with values based on published reports and news coverage of documented events — including South Lhonak Lake (the source of the October 2023 Sikkim flood), Chorabari Lake (linked to the 2013 Kedarnath disaster), and Parechu Lake (known for a 2005 outburst that affected Himachal Pradesh).
- **~780 synthetic lakes**, generated to fill out the dataset across all six regions, since real, labeled GLOF data at this scale does not exist publicly.

Synthetic risk labels are generated from a weighted score based on lake area, glacier retreat rate, slope, rainfall, distance from glacier, and seismic activity. This introduces an important limitation: since the label itself is built from the same features later used to train the model, a model trained and tested only on synthetic data will look artificially strong, because it is partly just learning to reproduce the formula that created its own labels.

To address this, the 10 real lakes are held out completely from training, validation, and model selection. They are only used afterward as a genuine test of whether the model generalizes beyond the synthetic formula. The current model correctly classifies 8 of the 10 real lakes. Since this evaluation is based on only ten samples, it should be treated as an initial indication of generalization rather than a definitive measure of real-world performance.

## How it works

1. `generate_dataset.py` builds `glof_dataset.csv` from the real lake list plus the synthetic generation logic described above.
2. `GLOF_Risk_Prediction.ipynb` loads the dataset, cleans it (handles missing values and duplicates), explores it, and trains four models — Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting — comparing them on a validation split. The best model by F1 score is evaluated on a held-out synthetic test set, and separately on the 10 real lakes, then saved along with the scaler and feature list.
3. `app.py` is a Streamlit dashboard that loads the saved model and lets you enter lake parameters to get a risk prediction, view all lakes on a map, see risk concentration on a heatmap, view the illustrative growth estimate, inspect SHAP feature explanations, and see example alerts.

## Limitations

- Synthetic labels are formula-derived, not observed outcomes, so overall model metrics on synthetic data should not be read as real-world accuracy.
- Only 10 real, labeled lakes exist in the dataset. That is not enough to confidently validate the model on real events, and the 80% figure above should be treated with that caveat.
- No real satellite imagery, remote sensing, or time-series monitoring is used anywhere in the pipeline yet.
- The model saved by the notebook is not fixed to one algorithm. Random Forest, Gradient Boosting, Logistic Regression, and Decision Tree are all trained and compared, and whichever scores best on validation gets saved. This selection can change if the code or data changes. With the same code and same dataset, results are reproducible thanks to fixed random seeds — but always check which model actually got saved (`glof_model.joblib`) rather than assuming it matches an earlier run or an earlier version of this document.
- Feature set is limited to geometric and climatic variables. It does not include some of the factors glaciology research considers important for GLOF risk, such as moraine dam composition or lake bathymetry, mainly because that data was not available at the scale needed here.

## Setup

```
pip install -r requirements.txt
python generate_dataset.py
jupyter notebook GLOF_Risk_Prediction.ipynb   # run all cells to train and save the model
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Builds the dataset from real lake records and synthetic generation |
| `GLOF_Risk_Prediction.ipynb` | Data cleaning, EDA, model training, and evaluation |
| `app.py` | Streamlit dashboard for prediction and visualization |
| `glof_model.joblib` | Trained model |
| `glof_scaler.joblib` | Feature scaler (used only for Logistic Regression) |
| `glof_feature_cols.joblib` | Feature column order expected by the model |
| `glof_region_map.joblib` | Mapping between region names and encoded values |
| `region_polygons.pkl` | State/UT boundary polygons used to place synthetic lakes realistically |

## Future work

- Replace or supplement synthetic labels with real hazard assessments where available, such as inventories from ICIMOD or published glacial lake hazard rankings.
- Integrate actual satellite imagery (Sentinel Hub or Google Earth Engine) for the growth estimate page, instead of the current assumed growth rate.
- Add real time-series data if a monitoring source becomes available, rather than the current illustrative trend chart.
- Expand the real-lake set as more documented events and lake surveys become accessible, to make the held-out evaluation more statistically meaningful.

## References

- South Lhonak Lake and the October 2023 Sikkim GLOF event
- Chorabari Lake and the June 2013 Kedarnath disaster
- Parechu Lake outburst, 2005, Himachal Pradesh


