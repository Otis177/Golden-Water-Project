import os
import pandas as pd
import xarray as xr
import numpy as np

"""Imports all libraries"""

def csv_to_panda_func(*filePaths):
    """
    Takes any number of file paths and loads each one into a pandas dataframe.
    Returns a list of dataframes in the same order as the file paths provided.
    """
    dataframes = []
    for filePath in filePaths:
        df = pd.read_csv(filePath)
        dataframes.append(df)
    return dataframes


def daily_diversions_to_weekly(df):
    """
    Converts a daily diversion dataframe into weekly totals using ISO calendar weeks.

    Steps:
        1. Parses the 'Date' column from string format ('January 1, 2001') into datetime
        2. Extracts ISO year and ISO week (used to match the weather data's week definition)
        3. Sums daily diversion values within each (year, week) group
        4. Drops 2003 entirely — that year has known corrupted data

    Returns a dataframe with columns: year, week, usage
    """
    # Parse date strings into datetime objects using the known format
    df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y')

    # Extract ISO calendar year and week — must use isocalendar() together
    # to avoid mismatches at year boundaries (e.g. Dec 31 being ISO week 1 of next year)
    iso = df['Date'].dt.isocalendar()
    df['year'] = iso.year.astype(int)
    df['week'] = iso.week.astype(int)

    # Sum daily values within each week — diversions are a volume so summing is correct
    dfWeekly = df.groupby(['year', 'week'])['Total Diverted Raw + Seasonal AF'].sum().reset_index()
    dfWeekly.rename(columns={'Total Diverted Raw + Seasonal AF': 'usage'}, inplace=True)

    # Drop 2003 — data is corrupted and would introduce bad lag values around that gap
    dfWeekly = dfWeekly[dfWeekly['year'] != 2003]

    return dfWeekly


def average_data(df, varName):
    """
    Averages the 6 weather grid cell columns into a single regional mean per week.

    The weather CSVs contain one column per grid cell (named as lat_lon coordinate pairs)
    representing different spatial points within the Golden, CO bounding box.
    Averaging across them gives one representative value for the region per week.

    Returns a dataframe with columns: year, week, {varName}
    """
    # Select all columns except year and week — these are the grid cell value columns
    gridCols = [col for col in df.columns if col not in ['year', 'week']]

    # Average across all grid cell columns for each row (axis=1 = across columns)
    df[varName] = df[gridCols].mean(axis=1)

    # Return only the columns needed for merging
    return df[['year', 'week', varName]]


def combine_dataframes(dataframeList):
    """
    Merges a list of weekly dataframes into one master dataframe on (year, week).

    Starts from index 1 (weekly diversions) and merges each weather dataframe in.
    Index 0 (sector monthly data) is intentionally skipped — it has different
    time resolution and is not used in the weekly model.

    Uses inner join so only weeks present in ALL datasets are kept.
    Returns the merged master dataframe.
    """
    # Start with the diversions dataframe as the base
    combinedDF = dataframeList[1]

    # Merge each subsequent dataframe (weather variables) onto the base
    for df in dataframeList[2:]:
        combinedDF = pd.merge(combinedDF, df, on=['year', 'week'], how='inner')

    return combinedDF


def lagged_data_maker(df, varName, weeksLagged):
    """
    Creates a new lagged column for a given weather variable.

    A lag of N means: what was the value of this variable N weeks ago?
    For example, pr_lag2 in week 10 contains the precipitation value from week 8.

    Rows that cannot be validly lagged are filled with the sentinel value -9999:
        - The first {weeksLagged} rows of the dataset (no prior data exists)
        - The first {weeksLagged} weeks of 2004 (the row after the 2003 data gap,
          so .shift() would incorrectly pull from 2002 across the gap)

    These sentinel rows are later removed by incompatible_data_remover().

    Returns the dataframe with a new column named {varName}_lag{weeksLagged}
    """
    newCol = f"{varName}_lag{weeksLagged}"

    # Sort by year then week to ensure shift() moves in the right direction
    df = df.sort_values(by=['year', 'week']).reset_index(drop=True)

    # Shift the column down by weeksLagged rows — NaN appears at the top
    df[newCol] = df[varName].shift(weeksLagged)

    # Fill NaN values (start of data) with sentinel value
    df[newCol] = df[newCol].fillna(-9999)

    # Also flag the first weeksLagged weeks of 2004 as invalid.
    # Without this, those rows would contain values shifted across the 2003 gap,
    # meaning they would incorrectly reflect 2002 conditions rather than 2003.
    badRows = (df['year'] == 2004) & (df['week'] <= weeksLagged)
    df.loc[badRows, newCol] = -9999

    return df


def inconpadable_data_remover(df):
    """
    Removes any row that contains the sentinel value -9999 in any column.

    The sentinel value -9999 is used to flag rows where lagged data is invalid —
    either because there is no prior data to lag from (start of dataset or after
    the 2003 gap). These rows must be removed before modeling.

    Returns the cleaned dataframe with reset index.
    """
    # Create a boolean mask: True where any column in that row equals -9999
    mask = (df == -9999).any(axis=1)

    # Keep only rows where the mask is False (no sentinel values)
    return df[~mask].reset_index(drop=True)


def week_53_remover(df):
    """
    Removes all rows where week == 53.

    Some ISO calendar years have 53 weeks instead of 52. These extra rows are
    inconsistent across years and would cause problems with the rolling window
    wrapping logic in window_maker(). Removing them keeps the week structure
    uniform at 52 weeks per year.

    Returns the cleaned dataframe with reset index.
    """
    return df[df['week'] != 53].reset_index(drop=True)


def window_maker(df, windowSize):
    """
    Builds a dictionary mapping each week number to a subset of the dataframe
    containing all rows within a rolling window of that week.

    Why this is needed: if we train a model on week 23 data only, we have at most
    one data point per year (~23 total). That is too few to train reliably. By
    borrowing neighboring weeks (e.g. weeks 20-26 for week 23), we give the model
    more rows to learn from, under the assumption that nearby weeks behave similarly.

    The window wraps around year boundaries using modulo arithmetic — so week 1's
    window includes weeks 51 and 52 from the end of the year, which makes
    climatological sense since late December and early January are similar.

    This is pre-computed once before the bootstrap loop so pandas doesnt have
    to re-filter the dataframe 1000 times per week.

    Returns a dictionary: {week_number: dataframe_subset}
    """
    # Use 52 as the max week to keep wrapping consistent across all years
    maxWeek = 52

    windowMap = {}
    for week in range(1, maxWeek + 1):
        # Generate the list of week numbers in this window, wrapping around 52→1
        # Example: week=1, windowSize=2 → [51, 52, 1, 2, 3]
        windowWeeks = [(w - 1) % maxWeek + 1 for w in range(week - windowSize, week + windowSize + 1)]

        # Filter the dataframe to only rows whose week falls in this window
        windowMap[week] = df[df['week'].isin(windowWeeks)].copy()

    return windowMap

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
        #print(f"Year: {year}, Total Precip: {group['pr'].sum()/len(group)}")
        yearPrTotals[year] = group['pr'].sum()/len(group)
    del yearPrTotals[2025]  # Remove 2025 from the dictionary as it is not a full year
    #print(yearPrTotals)
    yearPrTotals = dict(sorted(yearPrTotals.items(), key=lambda item: item[1]))
    #print("\n" + str(yearPrTotals))
    yearTypes['Dry'] = dict(list(yearPrTotals.items())[:8])
    yearTypes['Wet'] = dict(list(yearPrTotals.items())[-8:])
    yearTypes['Normal'] = dict(list(yearPrTotals.items())[8:-8])
    yearTypes['Random'] = dict(list(yearPrTotals.items()))
    #print("\n" + str(yearTypes))
    return yearTypes
