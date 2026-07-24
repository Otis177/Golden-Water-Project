import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import r2_score
from CSV_file_merger_2 import window_maker
from ridge_regression_2 import get_ridge_model, get_best_ridge_model
from lasso_regression_2 import get_lasso_model, get_best_lasso_model

"""Imports all libraries"""

def run_bootstrap(df, modelType, featureNames, targetName, nIterations, nTrain, windowSize, alpha = 1.0, restrictedTestYears=None):
    """
    Runs a bootstrap resampling loop over all weeks of the year using a given model.

    For each iteration:
        1. Randomly selects nTrain years for training, the rest become the test set
        2. For each of the 52 weeks, filters the data to a rolling window around that week
        3. Fits the model on the training rows and evaluates on the test rows
        4. Records train R² and test R² for that week and iteration

    After nIterations, returns a dataframe of all results which can be passed
    directly to the plotting function to generate the skill boxplot.

    Parameters:
        df          : the cleaned master dataframe with all features and target
        modelType   : string key matching an entry in modelTypesList (e.g. 'Ridge')
        featureNames: list of column names to use as X (weather vars and their lags)
        targetName  : name of the column to use as Y (e.g. 'usage')
        nIterations : number of bootstrap iterations to run
        nTrain      : number of years to use for training in each iteration
        windowSize  : number of weeks on each side to include in the rolling window

    Returns:
        results dataframe with columns: iteration, week, train_r2, test_r2
    """

    # ── MODEL REGISTRY ────────────────────────────────────────────────────────
    # Maps model name strings to their factory functions.
    # Each factory function returns a fresh, unfitted model instance when called.
    # A fresh model is needed every iteration to avoid reusing a previously fitted model.
    # OTIS: Add new models here as you build them — e.g. "XGBoost": get_xgboost_model
    modelTypesList = {
        "Ridge": lambda week: get_ridge_model(alpha=alpha),
        "Lasso": lambda week: get_lasso_model(alpha=alpha),
        "BestRidge": lambda week: get_best_ridge_model(week),
        "BestLasso": lambda week: get_best_lasso_model(week),
        "MLR": lambda week: get_ridge_model(alpha=0.0)
    }

    # Validate that the requested model type exists in the registry
    if modelType not in modelTypesList:
        raise ValueError(f"Unknown modelType '{modelType}'. Available: {list(modelTypesList.keys())}")

    # ── SETUP ─────────────────────────────────────────────────────────────────
    results = []

    # Pre-build the window map once before the loop — this is a dictionary mapping
    # each week number to the subset of rows in its rolling window.
    # Done here so pandas doesnt re-filter the dataframe on every iteration.
    print(f"  Building window map with windowSize={windowSize}...")
    windowMap = window_maker(df, windowSize)

    # Get all unique years available in the data for random sampling
    allYears = df['year'].unique()
    if (restrictedTestYears is None) or (len(restrictedTestYears) == 0):
        restrictedTestYears = allYears
    maxWeek = int(df['week'].max())

    print(f"  Starting {nIterations} bootstrap iterations across {maxWeek} weeks...")
    print(f"  Training on {nTrain} years per iteration, testing on {len(allYears) - nTrain} years.")

    # ── BOOTSTRAP LOOP ────────────────────────────────────────────────────────
    for i in range(nIterations):

        # Print progress every 100 iterations so we know the model is still running
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1} / {nIterations} iterations...")

        # Randomly select nTrain years for training without replacement.
        # The remaining years automatically become the test set
        #trainYears = np.random.choice(restrictedTestYears, size=nTrain, replace=False)
        #change here to have the training years to be restricted
        trainYears = np.random.choice(allYears, size=nTrain, replace=False) 
        testYears = np.setdiff1d(restrictedTestYears, trainYears)

        # ── WEEK LOOP ─────────────────────────────────────────────────────────
        for week in range(1, maxWeek + 1):

            # Retrieve the pre-built window subset for this week
            windowDF = windowMap[week]

            # Split the window data into train and test sets by year
            trainDF = windowDF[windowDF['year'].isin(trainYears)]
            testDF = windowDF[windowDF['year'].isin(testYears)]

            # Skip this week if there isnt enough data to train or test on.
            # This can happen for edge weeks or when the window is very small.
            if len(trainDF) < 5 or len(testDF) < 1:
                continue

            # Extract feature matrix (X) and target vector (y) for train and test
            xTrain = trainDF[featureNames].values
            yTrain = trainDF[targetName].values
            xTest = testDF[featureNames].values
            yTest = testDF[targetName].values

            # ── MODEL FITTING ─────────────────────────────────────────────────
            # Create a fresh model instance for this iteration using the factory function.
            # This ensures no state carries over from the previous iteration.
            model = modelTypesList[modelType](week)

            # Fit the model on training data and predict on test data
            yPred, trainR2 = model.fit(xTrain, yTrain), model.score(xTrain, yTrain)
            yPred = model.predict(xTest)

            # Compute out-of-sample R² by comparing predictions to actual test values
            testR2 = r2_score(yTest, yPred)

            # Store the result for this iteration and week
            results.append({
                'iteration': i,
                'week': week,
                'train_r2': trainR2,
                'test_r2': testR2
            })

    print(f"  Bootstrap complete. Total result rows: {len(results)}")

    # Convert the list of result dicts into a dataframe and return
    return pd.DataFrame(results)
