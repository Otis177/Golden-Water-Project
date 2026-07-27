# Golden-Water-Project

**INTRODUCTION**

This project builds and compares three statistical models, Multiple Linear Regression, Lasso Regression, and Ridge Regression, that predict weekly water demand for the City of Golden Colorado using weather variables including precipitation, temperature, vapor pressure deficit, and potential evapotranspiration. The codebase is modular by design, making it easy to add new data sources, test new model types, and tweak parameters without restructuring the pipeline.

The overall goal is to create a scientifically rigorous framework that validates whether Golden's 2019 water conservation measures actually reduced demand beyond what weather conditions alone would predict, and to deliver a model Golden can use operationally for years to come. The code in this repo specifically focuses on comparing statistical model types to determine which is optimal for the final model. Additional supporting code such as visualizations explaining model differences and a look at how different sectors use water month by month can be found in the eximplifying differences in models folder.

**REPOSITORY STRUCTURE**

Golden-Water-Project/
│
├── codebase 2/                        # Main modeling pipeline
│   ├── main_2.py                      # Entry point — orchestrates the full pipeline
│   ├── bootstrap_2.py                 # Bootstrap resampling loop
│   ├── CSV_file_merger_2.py           # Data loading, cleaning, and feature engineering
│   ├── ridge_regression_2.py          # Ridge regression model definition
│   ├── lasso_regression_2.py          # Lasso regression model definition
│   ├── data_plotter_2.py              # Skill boxplot and comparison plot functions
│   ├── find_best_alpha.py             # Alpha tuning analysis for Ridge and Lasso
│   ├── catigorize_years.py            # Wet/dry/normal year classification
│   ├── year_type_comparer.py          # Bar chart comparing model skill by year type
│   └── plot_daily_diversions.py       # Average daily diversion curve plot
│
├── data/                              # Input data files
│   ├── daily_diversions_2001-2024.csv # Daily water diversion measurements from Golden
│   ├── pr_weekly_resampled.csv        # Weekly precipitation (GridMET)
│   ├── pet_weekly_resampled.csv       # Weekly potential evapotranspiration (GridMET)
│   ├── vpd_weekly_resampled.csv       # Weekly vapor pressure deficit (GridMET)
│   ├── tmmx_weekly_resampled.csv      # Weekly maximum temperature (GridMET)
│   └── sector_monthly_use_long.csv    # Monthly water use by customer sector
│
├── Find best alpha/                   # Alpha tuning outputs
│   ├── Ridge_Regression/              # Ridge alpha analysis graphs and best values CSV
│   └── Lasso_Regression/              # Lasso alpha analysis graphs and best values CSV
│
├── Output data/                       # All model outputs
│   ├── Ridge/                         # Ridge regression results and graphs
│   ├── Lasso/                         # Lasso regression results and graphs
│   ├── BestRidge/                     # Best-alpha Ridge results by year type
│   ├── BestLasso/                     # Best-alpha Lasso results by year type
│   ├── MLR/                           # Multiple linear regression results and graphs
│   └── *.png                          # Combined comparison plots
│
├── eximplifying differences in models/ # Visualizations explaining model differences
│   ├── use_by_sector_visualization.py  # Monthly water use by sector
│   └── ridge_regression_visualization.py # Ridge vs MLR visual explainer
│
├── RandomForestModel.py               # Mentor's original Random Forest implementation
├── GoldenDataAnalysisFunctions.py     # Mentor's original data processing functions
└── README.md                          # This file

**SETUP AND INSTALLATION**

### Requirements
Python 3.8 or higher is required. Install all dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### File Paths
The codebase uses hardcoded file paths pointing to C:\Users\eo\Downloads\Golden project\. Before running anything you will need to update these paths in each file to match your local directory structure. The relevant path variables are near the top of main_2.py and in the file path dictionaries in find_best_alpha.py and year_type_comparer.py.

### Alpha CSV Files
ridge_regression_2.py and lasso_regression_2.py load best-alpha CSV files at import time. Make sure the paths to Best_Alpha_Values_Ridge.csv and Best_Alpha_Values_Lasso.csv in those files point to your local copies of the Find best alpha folder before running any Best model.

**How to Run**

Run files in this order:

1. main_2.py — runs the full pipeline. Configure model type, alpha, number of iterations, and year type at the top of the file before running. All other files are called automatically. 

2. find_best_alpha.py — run this separately to generate alpha tuning graphs and best alpha CSVs. Must be run before using BestRidge or BestLasso model types in main_2.py. In order to run this you must have previously ran main_2.py for each alpha value you wish to test.

3. year_type_comparer.py — run after you have R² CSV outputs for all three model types across all four year types.

4. plot_daily_diversions.py — standalone script, can be run at any time independently.

The files in eximplifying differences in models/ are also standalone and can be run independently.

---

**Data Sources**

| File | Source |
|---|---|
| daily_diversions_2001-2024.csv | City of Golden, CO — measured weekly water diversions |
| pr_weekly_resampled.csv | GridMET via NREL — weekly precipitation |
| pet_weekly_resampled.csv | GridMET via NREL — weekly potential evapotranspiration |
| vpd_weekly_resampled.csv | GridMET via NREL — weekly vapor pressure deficit |
| tmmx_weekly_resampled.csv | GridMET via NREL — weekly maximum temperature |
| sector_monthly_use_long.csv | City of Golden, CO — monthly water use by customer sector |

Weather data covers the Golden, CO area and was spatially averaged across 6 GridMET grid cells within the city bounding box. All other data sources were synthesized from these data files within other sections of the code.

---

**Authors and Acknowledgements**

Otis Halley Gotway — primary developer, RECCS 2026 intern
Nels Bjarke — research mentor, Western Water Assessment / NREL

This project was conducted through the [RECCS program](https://cires.colorado.edu/outreach/reccs) at the University of Colorado Boulder in partnership with the Western Water Assessment, a NOAA RISA team at CIRES.

Data provided by the City of Golden, CO. Weather data sourced from [GridMET](https://www.climatologylab.org/gridmet.html).

Funded in part by NSF Award EAR 1757930.
