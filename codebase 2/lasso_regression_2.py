import os
import pandas as pd
import xarray as xr
from sklearn.linear_model import Lasso

"""Imports all libraries"""


# Load the best alpha CSV for Lasso once when the module is imported.
bestLassoAlphaCSVDF = pd.read_csv("C:\\Users\\eo\\Downloads\\Golden project\\Find best alpha\\Lasso_Regression\\Best_Alpha_Values_Lasso.csv")

# Convert to a dictionary mapping week number to alpha value.
# CSV week column starts at 0 so we add 1 to align with the bootstrap's 1-52 week range.
bestLassoAlphaDict = dict(zip(bestLassoAlphaCSVDF['Week'] + 1, bestLassoAlphaCSVDF['Alpha']))

def get_lasso_model(alpha=1.0):
    """
    Creates and returns a configured Lasso Regression model instance.

    Lasso Regression is linear regression with L1 regularization — it adds a
    penalty proportional to the absolute value of the coefficients, which can lead to sparse models where some coefficients are exactly zero.

    This helps prevent overfitting to the small training sets used in each bootstrap iteration.

    The alpha parameter controls regularization strength:
        - Higher alpha = stronger penalty = simpler model, less likely to overfit
        - Lower alpha = weaker penalty = closer to plain linear regression
        - User defined alpha value is a sensible default starting point

    This function is registered in bootstrap.py's modelTypesList under "Lasso".
    bootstrap.py calls this function once per iteration to get a fresh unfitted model.

    Returns a lasso model instance ready to be fitted.
    """
    return Lasso(alpha=alpha)

def get_best_lasso_model(week):
    """
    Creates and returns a Lasso Regression model instance with the best alpha value for the given week.
    """
    alpha = bestLassoAlphaDict.get(week, 0.5)  # Default to 1.0 if week not found
    return Lasso(alpha=alpha)