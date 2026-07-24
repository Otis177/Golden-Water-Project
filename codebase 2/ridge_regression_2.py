import os
import pandas as pd
import xarray as xr
from sklearn.linear_model import Ridge


"""Imports all libraries"""

# Load the best alpha CSV once when the module is imported.
# This runs exactly once no matter how many times get_best_ridge_model() is called.
bestRidgeAlphaCSVDF = pd.read_csv("C:\\Users\\eo\\Downloads\\Golden project\\Find best alpha\\Ridge_Regression\\Best_Alpha_Values_Ridge.csv")

# Convert to a dictionary mapping week number to alpha value.
# CSV week column starts at 0 so we add 1 to align with the bootstrap's 1-52 week range.
bestRidgeAlphaDict = dict(zip(bestRidgeAlphaCSVDF['Week'] + 1, bestRidgeAlphaCSVDF['Alpha']))

def get_ridge_model(alpha=1.0):
    """
    Creates and returns a configured Ridge Regression model instance.

    Ridge Regression is linear regression with L2 regularization — it adds a
    penalty proportional to the square of the coefficients, which prevents the
    model from overfitting to the small training sets used in each bootstrap iteration.

    The alpha parameter controls regularization strength:
        - Higher alpha = stronger penalty = simpler model, less likely to overfit
        - Lower alpha = weaker penalty = closer to plain linear regression
        - User defined alpha value is a sensible default starting point

    This function is registered in bootstrap.py's modelTypesList under "Ridge".
    bootstrap.py calls this function once per iteration to get a fresh unfitted model.

    Returns a Ridge model instance ready to be fitted.
    """
    return Ridge(alpha=alpha)

def get_best_ridge_model(week):
    """
    Creates and returns a Ridge Regression model instance with the best alpha value for the given week.
    """
    alpha = bestRidgeAlphaDict.get(week, 10.0)  # Default to 10.0 if week not found
    return Ridge(alpha=alpha)
