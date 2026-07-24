import os
import pandas as pd
import xarray as xr
from CSV_file_merger_2 import *
from bootstrap_2 import run_bootstrap
from data_plotter_2 import plot_data, plot_all_data

"""Imports all libraries"""

def main():
    """
    Main entry point for the Golden, CO water diversion prediction pipeline.
    This function orchestrates the full workflow in order:
        1. Load all raw CSV files into pandas dataframes
        2. Convert daily diversions to weekly totals
        3. Average weather grid cells into a single regional value per variable
        4. Merge all dataframes into one master dataframe on (year, week)
        5. Generate lagged features for each weather variable
        6. Remove rows with sentinel values (-9999) or week 53
        7. Define feature names and run the bootstrap model
    """

    # ── DISPLAY SETTINGS ──────────────────────────────────────────────────────
    # OTIS: Comment these out if you dont want to see the full dataframe in the console
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    # ── CONFIGURATION ─────────────────────────────────────────────────────────
    multiInOneGraph = False  # If True, all models will be plotted on the same graph for comparison
    
    # Number of weeks to lag each weather variable back in time.
    # Each variable (pr, pet, vpd, tmmx) will get lag1, lag2, lag3, lag4 columns.
    nLags = 4

    #Year types to be used in the bootstrap model. Options are "Dry", "Wet", "Normal", "Random", "None"
    yearType = "None"

    # Number of bootstrap iterations to run
    nIterations = 100

    # Number of years to use for training in each bootstrap iteration.
    # The remaining years become the test set.
    nTrain = 17

    # Window size for the rolling week window used during model training.
    # A windowSize of 5 means each week borrows 5 weeks on either side,
    # giving the model more rows to train on per iteration.
    windowSize = 5

    # Which model to use — must match a key in the modelTypesList in bootstrap.py
    # do ridge snd set alpha to 0.00 for multiple linear regression as the two are mathematically identical
    # Ridge, Lasso, BestRidge, BestLasso,"MLR"
    modelType = "MLR"

    # Regularization strength for Ridge Regression and Lasso (higher = more regularization) if alpha is set to 0 the code will call it multiple linear regression as the two are mathmatically identical
    alpha = 10
    
    #Creates the output path for the CSV file to be saved to
    modelName = modelType
    if modelType == "Ridge" or modelType == "Lasso":
        modelName += f"_alpha_{alpha}"
    if modelName == "Ridge_alpha_0.0" or modelName == "Lasso_alpha_0.0" or modelName == "MLR":
        modelName = "Multiple_Linear_Regression"
        modelType = "MLR"
    csvTrainandTestOutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelType}\\Data\\Train and Test\\{modelName}_Train_and_Test_{nIterations}_iterations.csv"
    csvR2OutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelType}\\Data\\r2\\{modelName}_r2_{nIterations}_iterations.csv"
    if modelName[:4] == "Best":
        modelName += f"_best_alpha"
        csvTrainandTestOutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelType[4:]}\\Data\\Train and Test\\{modelName}_Train_and_Test_{nIterations}_iterations.csv"
        csvR2OutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelType[:4]}\\Data\\r2\\{modelName}_r2_{nIterations}_iterations.csv"
    if yearType != "None":
        csvTrainandTestOutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelType}\\Data\\Train and Test\\{modelName}_Train_and_Test_{nIterations}_iterations_trained_on_random_tested_on_{yearType}.csv"
        csvR2OutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelType}\\Data\\r2\\{modelName}_r2_{nIterations}_iterations_trained_on_random_tested_on_{yearType}.csv"
    
    # ── FILE PATHS ────────────────────────────────────────────────────────────
    # All input CSV files. Index 0 is sector monthly data (not used in modeling),
    # index 1 is daily diversions, indices 2-5 are weekly weather variables.
    filePaths = [
        'C:\\Users\\eo\\Downloads\\Golden project\\data\\sector_monthly_use_long.csv',       # [0] not used in model
        'C:\\Users\\eo\\Downloads\\Golden project\\data\\daily_diversions_2001-2024.csv',    # [1] target variable
        'C:\\Users\\eo\\Downloads\\Golden project\\data\\tmmx_weekly_resampled.csv',         # [2] max temperature
        'C:\\Users\\eo\\Downloads\\Golden project\\data\\pr_weekly_resampled.csv',           # [3] precipitation
        'C:\\Users\\eo\\Downloads\\Golden project\\data\\vpd_weekly_resampled.csv',          # [4] vapor pressure deficit
        'C:\\Users\\eo\\Downloads\\Golden project\\data\\pet_weekly_resampled.csv'           # [5] potential evapotranspiration
    ]

    # ── STEP 1: LOAD RAW CSVs ─────────────────────────────────────────────────
    print("Step 1: Loading raw CSV files...")
    dataframes = csv_to_panda_func(*filePaths)
    print(f"  Loaded {len(dataframes)} files successfully.")

    # ── STEP 2: CONVERT DAILY DIVERSIONS TO WEEKLY ───────────────────────────
    # Parses date strings, extracts ISO year/week, sums daily values to weekly totals.
    # Also drops 2003 which is known to have corrupted data.
    print("Step 2: Converting daily diversions to weekly totals...")
    dataframes[1] = daily_diversions_to_weekly(dataframes[1])
    print(f"  Diversions converted. Shape: {dataframes[1].shape}")

    # ── STEP 3: AVERAGE WEATHER GRID CELLS ───────────────────────────────────
    # Each weather CSV has 6 columns of grid cell values (lat_lon coordinate pairs).
    # We average across those 6 columns to get one regional value per week.
    print("Step 3: Averaging weather grid cells into regional means...")
    for i, varName in zip(range(2, len(dataframes)), ['tmmx', 'pr', 'vpd', 'pet']):
        dataframes[i] = average_data(dataframes[i], varName)
        print(f"  Averaged {varName}. Shape: {dataframes[i].shape}")

    # ── STEP 4: MERGE ALL DATAFRAMES ─────────────────────────────────────────
    # Inner merge on (year, week) so only weeks present in ALL datasets are kept.
    # Starts from index 1 (diversions) — index 0 (sector monthly) is intentionally excluded.
    print("Step 4: Merging all dataframes on (year, week)...")
    data = combine_dataframes(dataframes)
    print(f"  Master dataframe shape after merge: {data.shape}")

    # ── STEP 5: GENERATE LAGGED FEATURES ─────────────────────────────────────
    # For each weather variable, create lag1 through lag{nLags} columns.
    # Lags represent antecedent conditions (e.g. what the precipitation was 2 weeks ago).
    # Rows that cant be lagged (start of data, or after the 2003 gap) are filled with -9999.
    print(f"Step 5: Generating lagged features (lag 1 to {nLags}) for each weather variable...")
    for var in ['tmmx', 'pr', 'vpd', 'pet']:
        for lag in range(1, nLags + 1):
            data = lagged_data_maker(data, var, lag)
        print(f"  Lags created for {var}.")

    # ── STEP 6: CLEAN DATA ────────────────────────────────────────────────────
    # Remove rows flagged with the -9999 sentinel (invalid lag values).
    # Also remove week 53 rows, which appear in some ISO years but not consistently.
    print("Step 6: Removing invalid rows (-9999 sentinel values and week 53)...")
    rowsBefore = len(data)
    data = inconpadable_data_remover(data)
    data = week_53_remover(data)
    rowsAfter = len(data)
    print(f"  Removed {rowsBefore - rowsAfter} rows. Final shape: {data.shape}")
    print(f"  Years in data: {sorted(data['year'].unique())}")
    print(f"  Weeks in data: {sorted(data['week'].unique())}")

    # Save a debug copy of the cleaned master dataframe for inspection
    data.to_csv('C:\\Users\\eo\\Downloads\\Golden project\\debug_output.csv', index=False)
    print("  Debug CSV saved to Golden project folder.")

    # ── STEP 7: DEFINE FEATURES AND RUN MODEL ────────────────────────────────
    # featureNames = all columns except the identifiers (year, week) and the target (usage).
    # These are the X variables fed into the model.
    featureNames = [col for col in data.columns if col not in ['year', 'week', 'usage']]
    print(f"Step 7: Running {modelType} bootstrap model with {nIterations} iterations...")
    print(f"  Features ({len(featureNames)}): {featureNames}")
    #Get the year types to be used in the bootstrap model If needed
    restrictedTestYears = None
    if yearType != "None" and yearType != "Random":
        restrictedTestYears = list(catigorize_years(data)[yearType].keys())

    # Run the bootstrap and store results
    # results is a dataframe with columns: iteration, week, train_r2, test_r2
    if multiInOneGraph:
        # If multiInOneGraph is True, we will run all models and plot them on the same graph
        modelTypesToRun = ["MLR", "BestRidge", "BestLasso"]
        allResults = []
        for model in modelTypesToRun:
            results = run_bootstrap(data, model, featureNames, 'usage', nIterations, nTrain, windowSize, alpha=alpha, restrictedTestYears=restrictedTestYears)
            print(f"  Bootstrap for {model} complete. Results shape: {results.shape}")
                # Temporary diagnostic — remove after debugging
            if model == "BestLasso":
                print("NaN test_r2 by week:")
                print(results[results['test_r2'].isna()].groupby('week').size())
                print("\nExtreme test_r2 by week (abs > 10):")
                print(results[results['test_r2'].abs() > 10].groupby('week').size())
            allResults.append((model, results))
    
    else:
        results = run_bootstrap(data, modelType, featureNames, 'usage', nIterations, nTrain, windowSize, alpha=alpha, restrictedTestYears=restrictedTestYears)
        print(f"  Bootstrap for complete. Results shape: {results.shape}")
        print(results.head(1000))
    # ── STEP 8: SAVE DATA AS CSV ──────────────────────────────────────────────────
    print("Saving results to CSV...")
    if multiInOneGraph:
        for model, res in allResults:
            modelName = model
            if model == "Ridge" or model == "Lasso":
                modelName += f"_alpha_{alpha}"
            if modelName == "Ridge_alpha_0.0" or modelName == "Lasso_alpha_0.0" or modelName == "MLR":
                modelName = "Multiple_Linear_Regression"
            csvTrainandTestOutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{model}\\Data\\Train and Test\\{modelName}_Train_and_Test_{nIterations}_iterations.csv"
            csvR2OutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{model}\\Data\\r2\\{modelName}_r2_{nIterations}_iterations.csv"
            if modelName[:4] == "Best":
                modelName += f"_best_alpha"
                csvTrainandTestOutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{model[4:]}\\Data\\Train and Test\\{modelName}_Train_and_Test_{nIterations}_iterations.csv"
                csvR2OutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{model[:4]}\\Data\\r2\\{modelName}_r2_{nIterations}_iterations.csv"
            res.to_csv(csvTrainandTestOutputPath, index=False)
            print(f"Results for {model} saved to {csvTrainandTestOutputPath}")
    else:
        results.to_csv(csvTrainandTestOutputPath, index=False)

        print(f"Results saved to {csvTrainandTestOutputPath}")

    # ── STEP 9: PLOT RESULTS ──────────────────────────────────────────────────
    print("Plotting results...")
    if multiInOneGraph:
        combinedOutputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\combined_skill_boxplot_{nIterations}_iterations.png"
        plot_all_data(allResults, nIterations, outputPath=combinedOutputPath)
    
    else:
        plot_data(results, modelType, nIterations, alpha=alpha, r2CSVOutputPath=csvR2OutputPath, yearType=yearType)
    print("Plot saved to Graphs folder.")
    print("Complete")
main()
