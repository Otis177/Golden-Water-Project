import os
import pandas as pd
import xarray as xr
import numpy as np


def catigorize_years(df):
    """Returns a dictionary with 4 different dataframes each containing a different subsection of years
    1. Dry will contain the 8 dryest years
    2. Wet will contain the 8 wettest years
    3. Normal will contain the 7 years between
    4. Random will contain all years"""
    # Sort the dataframe by year and week to ensure proper grouping
    df = df.sort_values(by=['year', 'week']).reset_index(drop=True)

    yearPrTotals = {}
    yearTypes = {}
    # Calculate total precipitation per year
    for year, group in df.groupby('year'):
        print(f"Year: {year}, Total Precip: {group['pr'].sum()/len(group)}")
        yearPrTotals[year] = group['pr'].sum()/len(group)
    del yearPrTotals[2025]  # Remove 2025 from the dictionary as it is not a full year
    print(yearPrTotals)
    yearPrTotals = dict(sorted(yearPrTotals.items(), key=lambda item: item[1]))
    print("\n" + str(yearPrTotals))
    yearTypes['Dry'] = dict(list(yearPrTotals.items())[:8])
    yearTypes['Wet'] = dict(list(yearPrTotals.items())[-8:])
    yearTypes['Normal'] = dict(list(yearPrTotals.items())[8:-8])
    yearTypes['Random'] = dict(list(yearPrTotals.items()))
    print("\n" + str(yearTypes))
    return yearTypes


df = pd.read_csv('C:\\Users\\eo\\Downloads\\Golden project\\debug_output.csv')
catigorize_years(df)