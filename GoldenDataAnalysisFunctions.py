import os
import xarray as xr
import pandas as pd

variable_name_dict = {'pr':'Precipitation (mm)',
                      'vpd':'Vapor Pressure Deficit (kPa)',
                      'pet':'Potential ET (mm)',
                      'dem':'Water Demand (AF)',
                      'tmmx':'Temperature (K)'}

def extract_nc_data():
    def extract_netcdf_to_tabular(base_dir, output_dir, variables, years, bbox):
        """
        Extracts daily values for specific grid cells within a bounding box from
        annual netCDF files and saves them as CSVs.
        """
        # Create the output directory if it doesn't already exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract bounding box limits
        min_lon, max_lon = bbox['min_lon'], bbox['max_lon']
        min_lat, max_lat = bbox['min_lat'], bbox['max_lat']

        for year in years:
            for var in variables:
                # Construct the expected file path (e.g., base_dir/pet_2025.nc)
                file_name = f"{var}_{year}.nc"
                file_path = os.path.join(base_dir, file_name)

                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}. Skipping.")
                    continue

                print(f"Processing {var} for {year}...")

                # Open the dataset
                with xr.open_dataset(file_path) as ds:

                    # Check if the internal variable name matches the file prefix.
                    # If standard naming isn't used (e.g., 'potential_evapotranspiration' instead of 'pet'),
                    # this grabs the primary data variable dynamically.
                    actual_var = var if var in ds.data_vars else list(ds.data_vars)[0]

                    # Determine latitude orientation.
                    # If latitudes are stored descending (e.g., 90 to -90), the slice bounds must be reversed.
                    lat_vals = ds['lat'].values
                    if lat_vals[0] > lat_vals[-1]:
                        lat_slice = slice(max_lat, min_lat)
                    else:
                        lat_slice = slice(min_lat, max_lat)

                    #
                    # Slice the dataset to the bounding box
                    ds_subset = ds.sel(lon=slice(min_lon, max_lon), lat=lat_slice)

                    # Convert the subset xarray object into a 1D pandas DataFrame
                    df = ds_subset[actual_var].to_dataframe().reset_index()

                    # Create a uniform grid cell string identifier for column headers
                    # Rounding prevents highly precise float variations from creating distinct columns
                    df['grid_cell'] = df['lat'].round(4).astype(str) + '_' + df['lon'].round(4).astype(str)

                    # Account for differences in time dimension naming ('day' vs 'time')
                    time_col = 'day' if 'day' in df.columns else 'time'

                    # Pivot to the requested structure: rows = days, columns = grid cells
                    df_pivot = df.pivot(index=time_col, columns='grid_cell', values=actual_var)

                    # Save to CSV
                    out_filename = f"{var}_{year}_tabular.csv"
                    out_filepath = os.path.join(output_dir, out_filename)

                    df_pivot.to_csv(out_filepath)
                    print(f"Successfully saved {out_filepath}")


    # --- Configuration Section ---

    # File paths
    base_directory = "/Volumes/LivnehExt01/Gridmet/"  # Update with the path to your .nc files
    output_directory = "/Volumes/LivnehExt01/Gridmet/golden_tabular"  # Update with where you want the CSVs saved

    # Iteration lists
    variables_to_extract = [
                            # 'pet', 'vpd', 'pr',
                            'tmmx']
    years_to_extract = list(range(2000,2026))

    # Define bounding box coordinates (Decimal Degrees)
    bounding_box = {
        'min_lon': -105.2953,
        'max_lon': -105.1541,
        'min_lat': 39.7059,
        'max_lat': 39.8104
    }

    # Run the extraction
    extract_netcdf_to_tabular(
        base_dir=base_directory,
        output_dir=output_directory,
        variables=variables_to_extract,
        years=years_to_extract,
        bbox=bounding_box
    )

def merge_csv_files():
    import os
    import glob
    import pandas as pd

    def merge_annual_tabular_data(input_dir, output_dir, variables):
        """
        Merges annual CSV files for each variable into a single CSV file containing all years.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for var in variables:
            # Define the search pattern for the variable's CSV files
            # E.g., looks for "pet_*_tabular.csv"
            search_pattern = os.path.join(input_dir, f"{var}_*_tabular.csv")
            file_list = glob.glob(search_pattern)

            if not file_list:
                print(f"No files found for variable '{var}' matching pattern: {search_pattern}")
                continue

            print(f"Found {len(file_list)} files for '{var}'. Merging...")

            # List to hold each year's dataframe
            dataframes = []

            for file in file_list:
                # Read the CSV
                df = pd.read_csv(file)
                dataframes.append(df)

            # Concatenate all dataframes into one
            combined_df = pd.concat(dataframes, ignore_index=True)

            # Ensure the 'day' column is treated as a datetime object so we can sort chronologically
            if 'day' in combined_df.columns:
                combined_df['day'] = pd.to_datetime(combined_df['day'])
                combined_df = combined_df.sort_values(by='day')
                # Optional: convert back to string format 'YYYY-MM-DD' if preferred
                # combined_df['day'] = combined_df['day'].dt.strftime('%Y-%m-%d')
            elif 'time' in combined_df.columns:
                combined_df['time'] = pd.to_datetime(combined_df['time'])
                combined_df = combined_df.sort_values(by='time')

            # Define the output file path
            out_filename = f"{var}_all_years.csv"
            out_filepath = os.path.join(output_dir, out_filename)

            # Save the merged dataframe to CSV (without the numeric row index)
            combined_df.to_csv(out_filepath, index=False)
            print(f"Successfully saved merged data to: {out_filepath}\n")

    # --- Configuration Section ---

    # File paths
    # Update these to match where your previous script saved the files
    input_directory = "/Volumes/LivnehExt01/Gridmet/golden_tabular/"
    output_directory = "/Volumes/LivnehExt01/Gridmet/golden_tabular/merged"

    # The variables you want to merge
    variables_to_merge = [
                # 'pet', 'vpd', 'pr',
                    'tmmx']

    # Run the merge
    merge_annual_tabular_data(
        input_dir=input_directory,
        output_dir=output_directory,
        variables=variables_to_merge
    )

def weekly_resample_csvs():
    def resample_daily_to_weekly(input_dir, output_dir, variables):
        """
        Reads merged daily tabular data, extracts the year and week,
        and resamples grid-cell data based on variable-specific aggregation rules.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for var in variables:
            # Determine the correct aggregation method per variable
            if var in ['pet', 'ppt']:
                agg_method = 'sum'
            elif var == 'vpd':
                agg_method = 'mean'
            else:
                print(f"Warning: Unknown aggregation for '{var}'. Defaulting to 'mean'.")
                agg_method = 'mean'

            in_filename = f"{var}_all_years.csv"
            in_filepath = os.path.join(input_dir, in_filename)

            if not os.path.exists(in_filepath):
                print(f"File not found: {in_filepath}. Skipping.")
                continue

            print(f"Resampling '{var}' using '{agg_method}' aggregation...")

            # Load the merged daily data
            df = pd.read_csv(in_filepath)

            # Identify and convert the date column
            date_col = 'day' if 'day' in df.columns else 'time'
            df[date_col] = pd.to_datetime(df[date_col])

            # Extract the ISO year and week number
            # ISO calendars ensure consistent 7-day weeks even across Jan 1st
            df['year'] = df[date_col].dt.isocalendar().year
            df['week'] = df[date_col].dt.isocalendar().week

            # Drop the daily date column so it doesn't interfere with the numeric aggregation
            df_numeric = df.drop(columns=[date_col])

            # Group by the new 'year' and 'week' columns and apply the chosen math
            weekly_df = df_numeric.groupby(['year', 'week']).agg(agg_method)

            # Save to CSV.
            # By default, groupby leaves 'year' and 'week' as the MultiIndex,
            # so to_csv() will automatically make them the first two columns.
            out_filename = f"{var}_weekly_resampled.csv"
            out_filepath = os.path.join(output_dir, out_filename)

            weekly_df.to_csv(out_filepath)
            print(f"Successfully saved weekly data to: {out_filepath}\n")

    # --- Configuration Section ---

    # File paths
    input_directory = "/Volumes/LivnehExt01/Gridmet/golden_tabular/merged"  # Directory containing your merged daily files
    output_directory = "/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly"  # Directory where weekly files will be saved

    # Variables to process
    variables_to_process = [
        # 'pet', 'vpd', 'pr',
                            'tmmx'
                            ]

    # Run the resampling process
    resample_daily_to_weekly(
        input_dir=input_directory,
        output_dir=output_directory,
        variables=variables_to_process
    )

def plot_annualweekly_timeseries(input_dir = '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly',
                                var_ID = 'pr',
                                 rolling=False,
                                 window_size=5):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Pathformatting for the file
    pathform=f'/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/{var_ID}_weekly_resampled.csv'

    # Create the figure space
    fig,sax = plt.subplots(1,1,constrained_layout=True)

    #Load and subset to a single column for plotting right now
    df = pd.read_csv(pathform.format(var_ID=var_ID))
    df = df[df.columns[[0,1,-2]]]

    if rolling ==True:
        df[df.columns[-1]] = df[df.columns[-1]].rolling(window=window_size).mean()



    # Plot the values of the weekly variables with uncertainty bands
    sns.lineplot(data=df,
                 x='week',
                 y=df.columns[-1],
                 errorbar=('ci',95),
                 ax=sax)

    # Add title and labels
    sax.set_xlabel('Week of Year',fontsize='x-large')
    sax.set_ylabel(variable_name_dict[var_ID],fontsize='x-large')
    if rolling == True:
        sax.set_title(f'City of Golden - GridMET\n Rolling {window_size}-week'.format(window_size=window_size)+' '+variable_name_dict[var_ID],fontsize='x-large')
    else:
        sax.set_title('City of Golden - GridMET\n'+variable_name_dict[var_ID],fontsize='x-large')

    plt.show()

def plot_annualweekly_demand_timeseries(pathname = '/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/weekly_diversions.csv',
                                 rolling=False,
                                 window_size=5):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd


    # Create the figure space
    fig,sax = plt.subplots(1,1,constrained_layout=True)

    #Load and subset to a single column for plotting right now
    df = pd.read_csv(pathname)
    # df = df[df.columns[[0,1,-2]]]

    if rolling ==True:
        df[df.columns[-1]] = df[df.columns[-1]].rolling(window=window_size).mean()



    # Plot the values of the weekly variables with uncertainty bands
    sns.lineplot(data=df,#[(df.week>=26)&(df.week<=36)],
                 x='week',
                 y=df.columns[-1],
                 # hue='year',
                 errorbar=('ci',95),
                 ax=sax)

    # Add title and labels
    sax.set_xlabel('Week of Year',fontsize='x-large')
    sax.set_ylabel('Diversions (Acre-Feet)',fontsize='x-large')
    if rolling == True:
        sax.set_title(f'City of Golden - Diversions \n Rolling {window_size}-week'.format(window_size=window_size),fontsize='x-large')
    else:
        sax.set_title('City of Golden - Diversions',fontsize='x-large')

    plt.show()

def var_autocorrelation():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    def calculate_and_plot_smoothed_autocorrelation(filepath,
                                                    var_ID):
        # 1. Load the dataset
        df = pd.read_csv(filepath)

        # Identify grid cell columns (exclude 'year' and 'week')
        grid_columns = [col for col in df.columns if col not in ['year', 'week']]

        # Create a regional mean time-series
        df['regional_mean'] = df[grid_columns].mean(axis=1)

        # Sort chronologically to ensure the lags are perfectly sequential
        df = df.sort_values(by=['year', 'week']).reset_index(drop=True)

        lags = [1, 2, 3, 4, 5]
        n_obs = len(df)

        # Calculate the 95% confidence threshold of uncertainty
        uncertainty_threshold = 1.96 / np.sqrt(n_obs)

        # --- TASK 1: Overall Autocorrelation ---
        overall_acf = []

        for lag in lags:
            df[f'lag_{lag}'] = df['regional_mean'].shift(lag)
            # Drop NaNs created by the shift to calculate the Pearson correlation
            corr = df['regional_mean'].corr(df[f'lag_{lag}'])
            overall_acf.append(corr)

        # --- TASK 2: Autocorrelation by Week of the Year (Smoothed) ---
        weeks = sorted(df['week'].dropna().unique())
        max_week = int(df['week'].max())
        weekly_acf = {lag: [] for lag in lags}

        for week in weeks:
            # Define the +/- 3 week window (total 7 weeks)
            # We use modulo arithmetic so Week 1 grabs Week 52, etc.
            window_weeks = [(w - 1) % max_week + 1 for w in range(int(week) - 3, int(week) + 4)]

            # Filter data to only rows that belong to any of the 7 weeks in the window
            window_data = df[df['week'].isin(window_weeks)]

            for lag in lags:
                # Drop missing values to safely compute correlation for this specific window
                valid_data = window_data[['regional_mean', f'lag_{lag}']].dropna()

                if len(valid_data) > 1:
                    corr = valid_data['regional_mean'].corr(valid_data[f'lag_{lag}'])
                    weekly_acf[lag].append(corr)
                else:
                    weekly_acf[lag].append(np.nan)

        # --- PLOTTING ---

        # Plot 1: Overall Autocorrelation vs Lags
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(lags, overall_acf, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Autocorrelation')

        ax1.axhline(0, color='black', linewidth=1)
        ax1.axhline(uncertainty_threshold, color='red', linestyle='--', label='95% Uncertainty Threshold')
        ax1.axhline(-uncertainty_threshold, color='red', linestyle='--')

        ax1.set_xlabel('Lag (Weeks)')
        ax1.set_ylabel('Pearson Correlation Coefficient')
        ax1.set_title(f'Overall Weekly {variable_name_dict[var_ID]} Autocorrelation (Lags 1-5)')
        ax1.set_xticks(lags)
        ax1.set_ylim(0,1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        # fig1.savefig('overall_acf.png', dpi=300)
        # print("Saved 'overall_acf.png'")

        # Plot 2: Autocorrelation by Week of the Year (Smoothed)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, lag in enumerate(lags):
            ax2.plot(weeks, weekly_acf[lag], label=f'Lag {lag} Week', color=colors[i], linewidth=1.5)

        ax2.axhline(0, color='black', linewidth=1)

        # Add uncertainty threshold line
        ax2.axhline(uncertainty_threshold, color='black', linestyle=':', alpha=0.5, label='95% Threshold')
        ax2.axhline(-uncertainty_threshold, color='black', linestyle=':', alpha=0.5)

        ax2.set_xlabel('Target Week of the Year')
        ax2.set_ylabel('Pearson Correlation Coefficient')
        ax2.set_title('Smoothed Autocorrelation by Week (±3 Week Window)')
        ax2.set_xlim(1, max_week)

        # Prevent duplicate legend entries for the threshold line
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.25, 1))

        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        # fig2.savefig('weekly_acf_smoothed.png', dpi=300)
        # print("Saved 'weekly_acf_smoothed.png'")

    # Run the function
    var_ID='vpd'
    calculate_and_plot_smoothed_autocorrelation(f'/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/{var_ID}_weekly_resampled.csv',
                                                var_ID)

def resample_daily_to_weekly():
    import os
    import pandas as pd

    def resample_daily_to_weekly(input_dir, output_dir, variables):
        """
        Reads merged daily tabular data, extracts the year and week,
        and resamples grid-cell data based on variable-specific aggregation rules.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for var in variables:
            # Determine the correct aggregation method per variable
            if var in ['pet', 'pr']:
                agg_method = 'mean'
            elif var == 'vpd':
                agg_method = 'mean'
            else:
                print(f"Warning: Unknown aggregation for '{var}'. Defaulting to 'mean'.")
                agg_method = 'mean'

            in_filename = f"{var}_all_years.csv"
            in_filepath = os.path.join(input_dir, in_filename)

            if not os.path.exists(in_filepath):
                print(f"File not found: {in_filepath}. Skipping.")
                continue

            print(f"Resampling '{var}' using '{agg_method}' aggregation...")

            # Load the merged daily data
            df = pd.read_csv(in_filepath)

            # Identify and convert the date column
            date_col = 'day' if 'day' in df.columns else 'time'
            df[date_col] = pd.to_datetime(df[date_col])

            # Extract the ISO year and week number
            # ISO calendars ensure consistent 7-day weeks even across Jan 1st
            df['year'] = df[date_col].dt.isocalendar().year
            df['week'] = df[date_col].dt.isocalendar().week

            # Drop the daily date column so it doesn't interfere with the numeric aggregation
            df_numeric = df.drop(columns=[date_col])

            # Group by the new 'year' and 'week' columns and apply the chosen math
            weekly_df = df_numeric.groupby(['year', 'week']).agg(agg_method)

            # Save to CSV.
            # By default, groupby leaves 'year' and 'week' as the MultiIndex,
            # so to_csv() will automatically make them the first two columns.
            out_filename = f"{var}_weekly_resampled.csv"
            out_filepath = os.path.join(output_dir, out_filename)

            weekly_df.to_csv(out_filepath)
            print(f"Successfully saved weekly data to: {out_filepath}\n")

    # --- Configuration Section ---

    # File paths
    input_directory = "/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/"  # Directory containing your merged daily files
    output_directory = "/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly"  # Directory where weekly files will be saved

    # Variables to process
    variables_to_process = ['pet', 'vpd', 'pr']

    # Run the resampling process
    resample_daily_to_weekly(
        input_dir=input_directory,
        output_dir=output_directory,
        variables=variables_to_process
    )

def convert_csvs():
    import pandas as pd
    import calendar

    def convert_wide_to_long(input_filepath, output_filepath):
        """
        Converts a 'wide' format CSV of monthly usage data to a 'long' format.
        """
        # 1. Load the dataset
        df = pd.read_csv(input_filepath)

        # The very first column (Year) imports without a name in the header,
        # so pandas will name it 'Unnamed: 0'. We'll rename it to 'year'.
        df.rename(columns={df.columns[0]: 'year'}, inplace=True)

        # 2. Clean the column names
        # This strips out the 'kgal' string and any accidental leading/trailing spaces
        # (like the extra space in "February kgal ")
        df.columns = [col.replace('kgal', '').strip() for col in df.columns]

        # 3. Melt the DataFrame
        # Keep 'year' and 'Customer Category' as the identifier columns,
        # and "melt" the 12 month columns down into two new columns.
        id_vars = ['year', 'Customer Category']
        value_vars = [col for col in df.columns if col not in id_vars]

        df_long = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='month_str',
            value_name='value'
        )

        # 4. Map month names to integers (1-12)
        # The calendar module easily maps 'January' -> 1, 'February' -> 2, etc.
        month_map = {name: i for i, name in enumerate(calendar.month_name) if name}
        df_long['month'] = df_long['month_str'].map(month_map)

        # 5. Rename Customer Category and organize columns
        df_long.rename(columns={'Customer Category': 'customer class'}, inplace=True)

        # Reorder the columns to match your requested format exactly
        df_long = df_long[['year', 'month', 'customer class', 'value']]

        # Optional: Sort the dataframe chronologically so it's easier to read
        df_long = df_long.sort_values(by=['year', 'month', 'customer class']).reset_index(drop=True)

        # 6. Save the resulting long-format data
        df_long.to_csv(output_filepath, index=False)
        print(f"Successfully converted data to long format and saved to: {output_filepath}")

    # --- Configuration Section ---

    # File paths
    input_file = "/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/sector_monthly_use_wide_2019-2025.csv"
    output_file = "/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/sector_monthly_use_long.csv"

    # Run the function
    convert_wide_to_long(input_file, output_file)

def aggregate_demand():
    import pandas as pd

    def convert_daily_to_aggregated(input_file):
        """
        Reads a daily diversions CSV and generates two new files containing
        weekly and monthly sums formatted as [year, week, value] and [year, month, value].
        """
        # 1. Load the dataset
        df = pd.read_csv(input_file)

        # Identify the value column (which is the second column in the file)
        val_col = df.columns[1]

        # 2. Parse the string dates (e.g., 'January 1, 2001') into Datetime objects
        df['Date'] = pd.to_datetime(df['Date'])

        # Rename the diversion volume column to 'value'
        df = df.rename(columns={val_col: 'value'})

        # 3. Extract the date components needed for grouping
        # Standard calendar components for monthly grouping
        df['year_std'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month

        # ISO calendar components for weekly grouping to keep 7-day weeks from splitting across years
        df['year_iso'] = df['Date'].dt.isocalendar().year
        df['week'] = df['Date'].dt.isocalendar().week

        # 4. Weekly Resampling
        # Group by the ISO year and week, calculate the sum, and reset the index
        # to turn 'year' and 'week' back into standard columns
        weekly_df = df.groupby(['year_iso', 'week'])['value'].mean().reset_index()
        weekly_df.rename(columns={'year_iso': 'year'}, inplace=True)

        # 5. Monthly Resampling
        # Group by standard year and month, calculate the sum
        monthly_df = df.groupby(['year_std', 'month'])['value'].mean().reset_index()
        monthly_df.rename(columns={'year_std': 'year'}, inplace=True)

        # 6. Save both dataframes out to their own files without the integer index
        weekly_df.to_csv('/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/weekly_diversions.csv', index=False)
        print("Saved 'weekly_diversions.csv' with columns: [year, week, value]")

        monthly_df.to_csv('/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/monthly_diversions.csv', index=False)
        print("Saved 'monthly_diversions.csv' with columns: [year, month, value]")

    # Run the function on the provided file
    convert_daily_to_aggregated('/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/daily_diversions_2001-2024.csv')

def calc_regression_weekly_models():
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    def run_zscore_regression_and_plot(diversion_file, variable_files, train_years, test_years, pop_data, window=4):
        # 1. Load the target dataset (Weekly Diversions in Acre-Feet)
        df_target = pd.read_csv(diversion_file)
        df_target.rename(columns={'value': 'target_af'}, inplace=True)

        # Convert Acre-Feet to Gallons Per Capita (GPC)
        df_target['total_gallons'] = df_target['target_af'] * 325851
        df_target['population'] = df_target['year'].map(pop_data)
        df_target['target'] = df_target['total_gallons'] / df_target['population']

        sample_weeks = [12, 24, 36, 48]

        for var_name, var_file in variable_files.items():
            print(f"Processing Regression for {var_name.upper()}...")

            # Load the predictor dataset
            df_var = pd.read_csv(var_file)
            grid_cols = [c for c in df_var.columns if c not in ['year', 'week']]
            df_var['predictor'] = df_var[grid_cols].mean(axis=1)

            # Match Time Domains
            df = pd.merge(df_target, df_var[['year', 'week', 'predictor']],
                          on=['year', 'week'], how='inner')
            df = df.dropna(subset=['target'])
            max_week = int(df['week'].max())

            # 3. Train, Test, and Collect Data
            results = []
            scatter_data = {}

            for week in range(1, max_week + 1):
                window_weeks = [(w - 1) % max_week + 1 for w in range(int(week) - window, int(week) + window + 1)]
                window_df = df[df['week'].isin(window_weeks)].copy()

                # Split datasets
                train_df = window_df[(window_df['year'] >= train_years[0]) & (window_df['year'] <= train_years[1])]
                test_df = window_df[(window_df['year'] >= test_years[0]) & (window_df['year'] <= test_years[1])]

                X_train, y_train = train_df['predictor'].values, train_df['target'].values
                X_test, y_test = test_df['predictor'].values, test_df['target'].values

                if len(X_train) < 2 or len(X_test) < 1:
                    continue

                # Train regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(X_train, y_train)

                # Calculate testing skills
                y_pred = intercept + (slope * X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                train_r2 = r_value ** 2
                test_r2 = r2_score(y_test, y_pred)

                results.append({
                    'week': week, 'coefficient': slope, 'intercept': intercept,
                    'train_p_value': p_value, 'train_r2': train_r2,
                    'test_r2': test_r2, 'test_rmse': rmse
                })

                # Collect data for the 4-panel scatter figure
                if week in sample_weeks:
                    scatter_data[week] = {
                        'X_train': X_train, 'y_train': y_train,
                        'X_test': X_test, 'y_test': y_test,
                        'slope': slope, 'intercept': intercept
                    }

            results_df = pd.DataFrame(results)

            # --- PLOT 1: Training Skill (In-Sample R2) ---
            fig1, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(results_df['week'], results_df['train_r2'], color='#1f77b4',
                     marker='o', linestyle='-', label='Training $R^2$ (In-Sample)')

            ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Week of the Year')
            ax1.set_ylabel('$R^2$ Score')
            plt.title(
                f'Training Regression Skill: {var_name.upper()} predicting Weekly GPC\n(Training Years: {train_years[0]}-{train_years[1]})')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            fig1.tight_layout()
            # train_img = f'{var_name}_training_skill.png'
            # plt.savefig(train_img, dpi=300, bbox_inches='tight')
            # plt.close(fig1)
            # print(f"  Saved Training Skill Plot: {train_img}")

            # --- PLOT 2: Testing Skill (Out-of-Sample R2 and RMSE) ---
            fig2, ax1 = plt.subplots(figsize=(10, 6))

            # Testing R2
            ax1.plot(results_df['week'], results_df['test_r2'], color='#ff7f0e',
                     marker='s', linestyle='-', label='Testing $R^2$ (Out-of-Sample)')
            ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Week of the Year')
            ax1.set_ylabel('$R^2$ Score (Model Skill)')

            # Testing RMSE on Secondary Y-Axis
            ax2 = ax1.twinx()
            ax2.plot(results_df['week'], results_df['test_rmse'], color='#d62728',
                     marker='^', linestyle='-.', linewidth=2, label='Testing RMSE (Z-Score Units)')
            ax2.set_ylabel('RMSE (Standard Deviations)')

            plt.title(
                f'Testing Regression Skill: {var_name.upper()} predicting Weekly GPC\n(Testing Years: {test_years[0]}-{test_years[1]})')

            # Combine legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

            fig2.tight_layout()
            # test_img = f'{var_name}_testing_skill.png'
            # plt.savefig(test_img, dpi=300, bbox_inches='tight')
            # plt.close(fig2)
            # print(f"  Saved Testing Skill Plot: {test_img}")

            # --- PLOT 3: Four-Panel Scatter Plots ---
            fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for idx, w in enumerate(sample_weeks):
                if w not in scatter_data:
                    continue
                ax = axes[idx]
                d = scatter_data[w]

                # Scatter Points
                ax.scatter(d['X_train'], d['y_train'], color='#1f77b4', alpha=0.6,
                           label=f'Train ({train_years[0]}-{train_years[1]})')
                ax.scatter(d['X_test'], d['y_test'], color='#ff7f0e', alpha=0.8,
                           label=f'Test ({test_years[0]}-{test_years[1]})')

                # Regression Line
                x_min = min(d['X_train'].min(), d['X_test'].min())
                x_max = max(d['X_train'].max(), d['X_test'].max())
                x_line = np.linspace(x_min, x_max, 100)
                y_line = d['intercept'] + (d['slope'] * x_line)

                ax.plot(x_line, y_line, color='#d62728', linewidth=2, label='Regression Fit')

                ax.set_title(f'Week {w} Model')
                ax.set_xlabel(f'{var_name.upper()}')
                ax.set_ylabel('Withdrawal (GPC)')
                ax.grid(True, alpha=0.3)

                if idx == 0:
                    ax.legend(loc='upper right')

            fig3.tight_layout()
            plt.show()
            # scatter_img = f'{var_name}_scatter_panels.png'
            # plt.savefig(scatter_img, dpi=300)
            # plt.close(fig3)
            # print(f"  Saved Sample Scatters Plot: {scatter_img}\n")

    # --- Configuration ---
    train_period = (2001, 2019)
    test_period = (2020, 2024)
    anomaly_window = 5

    # Please fill in the exact values from your image table here
    # Format is Year: Population
    population_table = {
        2001: 17361,  # Replace with actual value from image
        2002: 17376,  # Replace with actual value from image
        2003: 17652,  # Replace with actual value from image
        2004: 17784,
        2005: 17721,
        2006: 17664,
        2007: 17701,
        2008: 17906,
        2009: 17965,
        2010: 18867,
        2011: 19351,
        2012: 19512,
        2013: 19610,
        2014: 20047,
        2015: 19440,
        2016: 20330,
        2017: 20533,
        2018: 20718,
        2019: 20842,
        2020: 20399,
        2021: 20702,
        2022: 20584,
        2023: 20242,
        2024: 20444  # Replace with actual value from image
    }

    # Files dictionary (Update paths if they are not in the same directory)
    # e.g., {'name_to_use': 'file_path.csv'}
    predictor_variables = {
        'pr': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pr_weekly_resampled.csv',
        'pet': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pet_weekly_resampled.csv',
        'vpd': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/vpd_weekly_resampled.csv'
    }
    # Run the function
    run_zscore_regression_and_plot(
        diversion_file='/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/weekly_diversions.csv',
        variable_files=predictor_variables,
        train_years=train_period,
        test_years=test_period,
        pop_data=population_table,
        window=anomaly_window
    )

def multilinear_regression():
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import itertools

    def run_mlr_selection_with_bar_plot(diversion_file, variable_files, train_years, test_years, pop_data, window=5):

        # 1. Load the Target Dataset (Weekly Diversions -> GPC)
        df_target = pd.read_csv(diversion_file)
        df_target.rename(columns={'value': 'target_af'}, inplace=True)
        df_target['total_gallons'] = df_target['target_af'] * 325851
        df_target['population'] = df_target['year'].map(pop_data)
        df_target['target'] = df_target['total_gallons'] / df_target['population']

        # 2. Load and Merge all Predictor Variables into a single DataFrame
        df = df_target.dropna(subset=['target']).copy()
        predictor_names = []

        for var_name, var_file in variable_files.items():
            print(f"Loading {var_name.upper()}...")
            df_var = pd.read_csv(var_file)
            grid_cols = [c for c in df_var.columns if c not in ['year', 'week']]

            # Calculate regional mean and label it with the variable name
            df_var[var_name] = df_var[grid_cols].mean(axis=1)

            # Merge into the main dataframe
            df = pd.merge(df, df_var[['year', 'week', var_name]], on=['year', 'week'], how='inner')
            predictor_names.append(var_name)

        max_week = int(df['week'].max())

        # 3. MLR Best Subset Selection and Testing
        results = []

        for week in range(1, max_week + 1):
            window_weeks = [(w - 1) % max_week + 1 for w in range(int(week) - window, int(week) + window + 1)]
            window_df = df[df['week'].isin(window_weeks)].copy()

            # Split into Train and Test
            train_df = window_df[(window_df['year'] >= train_years[0]) & (window_df['year'] <= train_years[1])]
            test_df = window_df[(window_df['year'] >= test_years[0]) & (window_df['year'] <= test_years[1])]

            if len(train_df) < 5 or len(test_df) < 1:
                continue

            y_train = train_df['target'].values
            y_test = test_df['target'].values

            # --- ROBUST SELECTION: Test all combinations of predictors ---
            best_aic = np.inf
            best_model = None
            best_features = None

            all_combinations = []
            for i in range(1, len(predictor_names) + 1):
                all_combinations.extend(itertools.combinations(predictor_names, i))

            for combo in all_combinations:
                features = list(combo)
                X_train_subset = train_df[features].values
                X_train_sm = sm.add_constant(X_train_subset)

                model = sm.OLS(y_train, X_train_sm).fit()

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_features = features

            # --- TESTING: Use the strictly selected best model on the Test Set ---
            X_test_subset = test_df[best_features].values
            X_test_sm = sm.add_constant(X_test_subset, has_constant='add')

            y_pred = best_model.predict(X_test_sm)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)

            results.append({
                'week': week,
                'selected_features': " + ".join(best_features),
                'num_features': len(best_features),
                'train_adj_r2': best_model.rsquared_adj,
                'train_aic': best_aic,
                'test_r2': test_r2,
                'test_rmse': rmse
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv('mlr_robust_selection_results.csv', index=False)

        # --- PLOT: Selected Variables Bar Chart with R2 Heights ---
        fig, ax = plt.subplots(figsize=(14, 7))

        # Identify all unique combinations of variables that were selected
        unique_combos = results_df['selected_features'].unique()

        # Generate a discrete colormap based on the number of unique combinations
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_combos)))
        color_map = dict(zip(unique_combos, colors))

        # Loop through each combination and plot its corresponding weeks as bars
        for combo in unique_combos:
            subset = results_df[results_df['selected_features'] == combo]
            ax.bar(subset['week'], subset['train_adj_r2'],
                   color=color_map[combo], edgecolor='black',
                   label=combo, width=0.8, zorder=3)

        # Overlay the out-of-sample Testing R2 as a continuous line for context
        ax.plot(results_df['week'], results_df['test_r2'], color='black',
                marker='o', linestyle='-', linewidth=2,
                label='Testing $R^2$ (Out-of-Sample)', zorder=4)

        # Formatting and aesthetics
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7, zorder=1)
        ax.set_xlabel('Week of the Year', fontsize=12)
        ax.set_ylabel('$R^2$ Score (Model Skill)', fontsize=12)
        ax.set_title(
            f'MLR Variable Selection and Model Skill by Week\n(Training Years: {train_years[0]}-{train_years[1]} | Testing Years: {test_years[0]}-{test_years[1]})',
            fontsize=14)

        ax.set_xlim(0, 53)
        ax.set_xticks(range(1, 53, 2))  # Show every other week to prevent x-axis crowding
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

        # Place legend cleanly outside the plot
        ax.legend(title='Selected Variables (Bar Color)', bbox_to_anchor=(1.01, 1), loc='upper left')

        fig.tight_layout()
        plt.show()
        plt.savefig('mlr_weekly_selection_bars.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved Weekly Variable Selection Plot: mlr_weekly_selection_bars.png\n")

    # --- Configuration Section ---

    train_period = (2001, 2015)
    test_period = (2016, 2024)
    anomaly_window = 0

    population_table = {
        2001: 17361, 2002: 17376, 2003: 17652, 2004: 17784,
        2005: 17721, 2006: 17664, 2007: 17701, 2008: 17906,
        2009: 17965, 2010: 18867, 2011: 19351, 2012: 19512,
        2013: 19610, 2014: 20047, 2015: 19440, 2016: 20330,
        2017: 20533, 2018: 20718, 2019: 20842, 2020: 20399,
        2021: 20702, 2022: 20584, 2023: 20242, 2024: 20444
    }

    predictor_variables = {
        'pr': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pr_weekly_resampled.csv',
        'pet': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pet_weekly_resampled.csv',
        'vpd': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/vpd_weekly_resampled.csv'
    }

    run_mlr_selection_with_bar_plot(
        diversion_file='/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/weekly_diversions.csv',
        variable_files=predictor_variables,
        train_years=train_period,
        test_years=test_period,
        pop_data=population_table,
        window=anomaly_window
    )


def bootstrap_mlr():
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import itertools
    import calendar

    def run_bootstrap_mlr_selection(diversion_file, variable_files, pop_data,
                                    window=5, n_iterations=50, n_train=17):

        # 1. Load the Target Dataset (Weekly Diversions -> GPC)
        df_target = pd.read_csv(diversion_file)
        df_target.rename(columns={'value': 'target_af'}, inplace=True)
        df_target['total_gallons'] = df_target['target_af'] * 325851
        df_target['population'] = df_target['year'].map(pop_data)
        df_target['target'] = df_target['total_gallons'] / df_target['population']

        # 2. Load and Merge Predictor Variables
        df = df_target.dropna(subset=['target']).copy()
        predictor_names = []

        for var_name, var_file in variable_files.items():
            print(f"Loading {var_name.upper()}...")
            df_var = pd.read_csv(var_file)
            grid_cols = [c for c in df_var.columns if c not in ['year', 'week']]
            df_var[var_name] = df_var[grid_cols].mean(axis=1)
            df = pd.merge(df, df_var[['year', 'week', var_name]], on=['year', 'week'], how='inner')
            predictor_names.append(var_name)

        all_years = df['year'].unique()
        max_week = int(df['week'].max())

        # Pre-calculate combinations
        all_combinations = []
        for i in range(1, len(predictor_names) + 1):
            all_combinations.extend(itertools.combinations(predictor_names, i))

        # Pre-extract rolling window data to save processing time
        window_data_map = {}
        for week in range(1, max_week + 1):
            window_weeks = [(w - 1) % max_week + 1 for w in range(int(week) - window, int(week) + window + 1)]
            window_data_map[week] = df[df['week'].isin(window_weeks)].copy()

        # 3. Bootstrap Resampling Loop
        print(f"Starting {n_iterations} bootstrap iterations...")
        results = []

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1} iterations...")

            train_years = np.random.choice(all_years, size=n_train, replace=False)
            test_years = np.setdiff1d(all_years, train_years)

            for week in range(1, max_week + 1):
                window_df = window_data_map[week]

                train_df = window_df[window_df['year'].isin(train_years)]
                test_df = window_df[window_df['year'].isin(test_years)]

                if len(train_df) < 5 or len(test_df) < 1:
                    continue

                y_train = train_df['target'].values
                y_test = test_df['target'].values

                best_aic = np.inf
                best_model = None
                best_features = None

                # Robust Selection
                for combo in all_combinations:
                    features = list(combo)
                    X_train_sm = sm.add_constant(train_df[features].values)
                    model = sm.OLS(y_train, X_train_sm).fit()

                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                        best_features = features

                # Out-of-sample Testing
                X_test_sm = sm.add_constant(test_df[best_features].values, has_constant='add')
                y_pred = best_model.predict(X_test_sm)
                test_r2 = r2_score(y_test, y_pred)

                results.append({
                    'iteration': i,
                    'week': week,
                    'selected_features': " + ".join(best_features),
                    'train_r2': best_model.rsquared,
                    'test_r2': test_r2
                })

        results_df = pd.DataFrame(results)

        # --- 4. Process Data for Plotting ---
        summary = []
        box_data = []
        median_test_skills = []

        for w in range(1, max_week + 1):
            w_data = results_df[results_df['week'] == w]

            train_r2_vals = w_data['train_r2'].values
            box_data.append(train_r2_vals)

            test_r2_vals = w_data['test_r2'].values
            test_r2_vals = test_r2_vals[test_r2_vals > -2.0]

            if len(test_r2_vals) > 0:
                median_test_skills.append(np.median(test_r2_vals))
            else:
                median_test_skills.append(np.nan)

            if not w_data.empty:
                counts = w_data['selected_features'].value_counts()
                dominant_feature = counts.index[0]
                pct = (counts.iloc[0] / len(w_data)) * 100
            else:
                dominant_feature = 'None'
                pct = 0

            summary.append({
                'week': w,
                'dominant_feature': dominant_feature,
                'pct': pct
            })

        summary_df = pd.DataFrame(summary)
        unique_combos = results_df['selected_features'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_combos)))
        color_map = dict(zip(unique_combos, colors))

        # --- 5. Generate Box and Whisker Plot ---
        fig, ax = plt.subplots(figsize=(20, 8))

        bp = ax.boxplot(box_data, positions=range(1, max_week + 1),
                        patch_artist=True, showfliers=False, widths=0.7)

        for idx, box in enumerate(bp['boxes']):
            w = idx + 1
            dom = summary_df.loc[summary_df['week'] == w, 'dominant_feature'].values[0]
            pct = summary_df.loc[summary_df['week'] == w, 'pct'].values[0]

            box.set_facecolor(color_map[dom])
            box.set_edgecolor('black')
            box.set_alpha(0.9)

            upper_whisker = bp['caps'][2 * idx + 1].get_ydata()[0]

            if pct > 0:
                ax.text(w, upper_whisker + 0.05, f'{pct:.0f}%',
                        ha='center', va='bottom', fontsize=9, rotation=45, color='black')

        ax.plot(range(1, max_week + 1), median_test_skills,
                marker='*', color='black', linestyle='None',
                markersize=10, zorder=5, label='Median Out-of-Sample Skill')

        ax.axhline(0, color='red', linestyle='--', alpha=0.7, zorder=0, label='Zero Skill Baseline')
        ax.set_xlabel('Week of the Year', fontsize=12, labelpad=30)  # Increased labelpad to make room for months
        ax.set_ylabel('$R^2$ Score', fontsize=12)
        ax.set_title(
            f'Bootstrapped Model Skill by Week ({n_iterations} Iterations)\nBoxes = In-Sample Training $R^2$ | Stars = Median Out-of-Sample Testing $R^2$',
            fontsize=15)

        # --- NEW: Map Weeks to Months and add Secondary X-Axis ---
        weeks_array = np.arange(1, max_week + 1)
        # Use a dummy non-leap year to calculate standard week-to-month mapping
        dummy_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta((weeks_array - 1) * 7, unit='D')
        month_series = dummy_dates.month

        month_centers = []
        month_labels = []

        for m in range(1, 13):
            weeks_in_m = weeks_array[month_series == m]
            if len(weeks_in_m) > 0:
                # Find the center point of the month for label placement
                month_centers.append(np.mean(weeks_in_m))
                month_labels.append(calendar.month_abbr[m])

                # Draw vertical line separating months (except before January)
                if m > 1:
                    boundary_week = weeks_in_m[0] - 0.5
                    ax.axvline(boundary_week, color='gray', linestyle=':', alpha=0.4, zorder=0)

        # Add the secondary axis for Month labels
        ax_months = ax.secondary_xaxis('bottom')
        ax_months.spines['bottom'].set_position(('outward', 30))  # Pushes this axis down below the week labels
        ax_months.set_xticks(month_centers)
        ax_months.set_xticklabels(month_labels, fontsize=12, fontweight='bold')
        ax_months.tick_params(axis='x', length=0)  # Hide the actual tick marks for a cleaner look

        # Create Custom Legend
        legend_elements = [Patch(facecolor=color_map[combo], edgecolor='black', label=combo) for combo in unique_combos]
        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=14,
                                      label='Median Out-of-Sample $R^2$'))
        ax.legend(handles=legend_elements, title='Dominant Selected Variables',
                  bbox_to_anchor=(1.01, 1), loc='upper left')

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_xticks(range(1, 53))

        plt.ylim(-1.5, 1.2)

        fig.tight_layout()
        plt.show()
        # plt.savefig('mlr_bootstrap_boxplot_with_months.png', dpi=300, bbox_inches='tight')
        # plt.close(fig)
        print("\nSaved Bootstrap Plot: mlr_bootstrap_boxplot_with_months.png")

    # --- Configuration Section ---
    anomaly_window = 1

    population_table = {
        2001: 17361, 2002: 17376, 2003: 17652, 2004: 17784,
        2005: 17721, 2006: 17664, 2007: 17701, 2008: 17906,
        2009: 17965, 2010: 18867, 2011: 19351, 2012: 19512,
        2013: 19610, 2014: 20047, 2015: 19440, 2016: 20330,
        2017: 20533, 2018: 20718, 2019: 20842, 2020: 20399,
        2021: 20702, 2022: 20584, 2023: 20242, 2024: 20444
    }

    predictor_variables = {
        'pr': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pr_weekly_resampled.csv',
        'pet': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pet_weekly_resampled.csv',
        'vpd': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/vpd_weekly_resampled.csv'
    }

    run_bootstrap_mlr_selection(
        diversion_file='/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/weekly_diversions.csv',
        variable_files=predictor_variables,
        pop_data=population_table,
        window=anomaly_window,
        n_iterations=1000,
        n_train=17
    )

def bootstrap_MLR_offsets():
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import itertools
    import calendar

    def run_lagged_bootstrap_mlr(diversion_file, variable_files, pop_data,
                                 offsets=[0, -1, -2], window=5, max_features=3,
                                 n_iterations=1000, n_train=17):

        # 1. Load the Target Dataset (Weekly Diversions -> GPC)
        df_target = pd.read_csv(diversion_file)
        df_target.rename(columns={'value': 'target_af'}, inplace=True)
        df_target['total_gallons'] = df_target['target_af'] * 325851
        df_target['population'] = df_target['year'].map(pop_data)
        df_target['target'] = df_target['total_gallons'] / df_target['population']

        # 2. Load, Generate Lags, and Merge Predictor Variables
        df = df_target.dropna(subset=['target']).copy()
        predictor_names = []

        for var_name, var_file in variable_files.items():
            print(f"Loading {var_name.upper()} and generating lags {offsets}...")
            df_var = pd.read_csv(var_file)
            grid_cols = [c for c in df_var.columns if c not in ['year', 'week']]
            df_var['regional_mean'] = df_var[grid_cols].mean(axis=1)

            # Ensure chronological order before shifting
            df_var.sort_values(['year', 'week'], inplace=True)

            cols_to_merge = ['year', 'week']

            # Generate the specified lags for this variable
            for offset in offsets:
                lag_num = abs(offset)
                col_name = f"{var_name}_lag{lag_num}"

                # offset=-1 -> shift(1) -> week 32 row receives week 31 data
                df_var[col_name] = df_var['regional_mean'].shift(-offset)

                cols_to_merge.append(col_name)
                predictor_names.append(col_name)

            # Merge all generated lags for this variable into the main dataframe
            df = pd.merge(df, df_var[cols_to_merge], on=['year', 'week'], how='inner')

        # Drop NaNs created at the edges of the dataset by the shifting process (e.g. Week 1 has no Week -1)
        df.dropna(subset=predictor_names, inplace=True)

        all_years = df['year'].unique()
        max_week = int(df['week'].max())

        # 3. Pre-calculate MLR combinations (Restricted to 'max_features' to prevent extreme overfitting)
        all_combinations = []
        for i in range(1, max_features + 1):
            all_combinations.extend(itertools.combinations(predictor_names, i))
        print(f"Testing {len(all_combinations)} possible MLR configurations per week.")

        # Pre-extract rolling window data to save processing time
        window_data_map = {}
        for week in range(1, max_week + 1):
            window_weeks = [(w - 1) % max_week + 1 for w in range(int(week) - window, int(week) + window + 1)]
            window_data_map[week] = df[df['week'].isin(window_weeks)].copy()

        # 4. Bootstrap Resampling Loop
        print(f"Starting {n_iterations} bootstrap iterations...")
        results = []

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1} iterations...")

            train_years = np.random.choice(all_years, size=n_train, replace=False)
            test_years = np.setdiff1d(all_years, train_years)

            for week in range(1, max_week + 1):
                window_df = window_data_map[week]

                train_df = window_df[window_df['year'].isin(train_years)]
                test_df = window_df[window_df['year'].isin(test_years)]

                if len(train_df) < 5 or len(test_df) < 1:
                    continue

                y_train = train_df['target'].values
                y_test = test_df['target'].values

                best_aic = np.inf
                best_model = None
                best_features = None

                # Robust Selection across the 129 combinations
                for combo in all_combinations:
                    features = list(combo)
                    X_train_sm = sm.add_constant(train_df[features].values)
                    model = sm.OLS(y_train, X_train_sm).fit()

                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                        best_features = features

                # Out-of-sample Testing
                X_test_sm = sm.add_constant(test_df[best_features].values, has_constant='add')
                y_pred = best_model.predict(X_test_sm)
                test_r2 = r2_score(y_test, y_pred)

                results.append({
                    'iteration': i,
                    'week': week,
                    'selected_features': " + ".join(best_features),
                    'train_r2': best_model.rsquared,
                    'test_r2': test_r2
                })

        results_df = pd.DataFrame(results)

        # --- 5. Process Data for Plotting ---
        summary = []
        box_data = []
        median_test_skills = []

        for w in range(1, max_week + 1):
            w_data = results_df[results_df['week'] == w]

            train_r2_vals = w_data['train_r2'].values
            box_data.append(train_r2_vals)

            test_r2_vals = w_data['test_r2'].values
            test_r2_vals = test_r2_vals[test_r2_vals > -2.0]

            if len(test_r2_vals) > 0:
                median_test_skills.append(np.median(test_r2_vals))
            else:
                median_test_skills.append(np.nan)

            if not w_data.empty:
                counts = w_data['selected_features'].value_counts()
                dominant_feature = counts.index[0]
                pct = (counts.iloc[0] / len(w_data)) * 100
            else:
                dominant_feature = 'None'
                pct = 0

            summary.append({
                'week': w,
                'dominant_feature': dominant_feature,
                'pct': pct
            })

        summary_df = pd.DataFrame(summary)

        # Generate colors ONLY for the combinations that actually dominated at least one week
        # (Otherwise the legend would try to display all 129 combinations!)
        unique_dom_combos = summary_df['dominant_feature'].unique()
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(unique_dom_combos)))
        color_map = dict(zip(unique_dom_combos, colors))

        # --- 6. Generate Box and Whisker Plot ---
        fig, ax = plt.subplots(figsize=(22, 9))

        bp = ax.boxplot(box_data, positions=range(1, max_week + 1),
                        patch_artist=True, showfliers=False, widths=0.7)

        for idx, box in enumerate(bp['boxes']):
            w = idx + 1
            dom = summary_df.loc[summary_df['week'] == w, 'dominant_feature'].values[0]
            pct = summary_df.loc[summary_df['week'] == w, 'pct'].values[0]

            box.set_facecolor(color_map[dom])
            box.set_edgecolor('black')
            box.set_alpha(0.9)

            upper_whisker = bp['caps'][2 * idx + 1].get_ydata()[0]

            if pct > 0:
                ax.text(w, upper_whisker + 0.05, f'{pct:.0f}%',
                        ha='center', va='bottom', fontsize=9, rotation=45, color='black')

        ax.plot(range(1, max_week + 1), median_test_skills,
                marker='*', color='black', linestyle='None',
                markersize=10, zorder=5, label='Median Out-of-Sample Skill')

        ax.axhline(0, color='red', linestyle='--', alpha=0.7, zorder=0, label='Zero Skill Baseline')
        ax.set_xlabel('Week of the Year', fontsize=12, labelpad=30)
        ax.set_ylabel('$R^2$ Score', fontsize=12)

        ax.set_title(
            f'Bootstrapped Model Skill w/ Antecedent Lags ({n_iterations} Iterations)\nAvailable Pool: {list(offsets)} wk lags | Max Variables per Model: {max_features}\nBoxes = In-Sample Training $R^2$ | Stars = Median Out-of-Sample Testing $R^2$',
            fontsize=15)

        # --- Map Weeks to Months and add Secondary X-Axis ---
        weeks_array = np.arange(1, max_week + 1)
        dummy_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta((weeks_array - 1) * 7, unit='D')
        month_series = dummy_dates.month

        month_centers = []
        month_labels = []

        for m in range(1, 13):
            weeks_in_m = weeks_array[month_series == m]
            if len(weeks_in_m) > 0:
                month_centers.append(np.mean(weeks_in_m))
                month_labels.append(calendar.month_abbr[m])

                if m > 1:
                    boundary_week = weeks_in_m[0] - 0.5
                    ax.axvline(boundary_week, color='gray', linestyle=':', alpha=0.4, zorder=0)

        ax_months = ax.secondary_xaxis('bottom')
        ax_months.spines['bottom'].set_position(('outward', 30))
        ax_months.set_xticks(month_centers)
        ax_months.set_xticklabels(month_labels, fontsize=12, fontweight='bold')
        ax_months.tick_params(axis='x', length=0)

        # Create Custom Legend (Only mapping colors for the combos that won a week)
        legend_elements = [Patch(facecolor=color_map[combo], edgecolor='black', label=combo) for combo in
                           unique_dom_combos]
        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=14,
                                      label='Median Out-of-Sample $R^2$'))

        ax.legend(handles=legend_elements, title='Dominant Selected Variables',
                  bbox_to_anchor=(1.01, 1), loc='upper left')

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_xticks(range(1, 53))

        plt.ylim(-1.5, 1.2)

        fig.tight_layout()
        plt.show()
        # plt.savefig('mlr_bootstrap_lagged.png', dpi=300, bbox_inches='tight')
        # plt.show()
        print("\nSaved Lagged Bootstrap Plot: mlr_bootstrap_lagged.png")

    # --- Configuration Section ---
    anomaly_window = 8

    population_table = {
        2001: 17361, 2002: 17376, 2003: 17652, 2004: 17784,
        2005: 17721, 2006: 17664, 2007: 17701, 2008: 17906,
        2009: 17965, 2010: 18867, 2011: 19351, 2012: 19512,
        2013: 19610, 2014: 20047, 2015: 19440, 2016: 20330,
        2017: 20533, 2018: 20718, 2019: 20842, 2020: 20399,
        2021: 20702, 2022: 20584, 2023: 20242, 2024: 20444
    }

    predictor_variables = {
        'pr': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pr_weekly_resampled.csv',
        'pet': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/pet_weekly_resampled.csv',
        'vpd': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/vpd_weekly_resampled.csv'
    }

    # Provide the global list of offsets to test for all variables
    lag_offsets = [0, -1, -2]

    run_lagged_bootstrap_mlr(
        diversion_file='/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/weekly_diversions.csv',
        variable_files=predictor_variables,
        pop_data=population_table,
        offsets=lag_offsets,
        window=anomaly_window,
        max_features=3,  # Sets the cap to 3 predictors per model maximum
        n_iterations=1000,
        n_train=17
    )

def plot_monthly_customerclass():
    import pandas as pd
    import matplotlib.pyplot as plt
    import calendar
    import numpy as np

    def plot_monthly_sector_use(filepath):
        # 1. Load the data
        df = pd.read_csv(filepath)

        # 2. Extract unique customer classes and sort the years
        classes = sorted(df['customer class'].unique())
        years = sorted(df['year'].unique())

        # Generate a list of month numbers (1-12) and their corresponding string names
        months = np.arange(1, 13)
        month_names = [calendar.month_abbr[i] for i in months]

        # 3. Create a color palette for the years (using 'viridis' colormap)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(years)))
        year_colors = dict(zip(years, colors))

        # 4. Set up the figure (4 rows, 2 columns for 8 total panels)
        fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=True)
        axes = axes.flatten()

        # 5. Loop through each sector and plot its data
        for i, c_class in enumerate(classes):
            if i >= len(axes):
                break  # Failsafe in case there are more than 8 classes

            ax = axes[i]
            class_data = df[df['customer class'] == c_class]

            # A) Plot the individual yearly lines
            for year in years:
                year_data = class_data[class_data['year'] == year].sort_values('month')

                if not year_data.empty:
                    # We only add the label once (on the first plot) so the legend doesn't duplicate
                    label_name = str(year) if i == 0 else ""

                    ax.plot(year_data['month'], year_data['value'],
                            marker='o', color=year_colors[year], alpha=0.6,
                            linewidth=1.5, markersize=5, label=label_name)

            # B) Calculate and plot the Average line
            mean_data = class_data.groupby('month')['value'].mean().reset_index()

            if not mean_data.empty:
                label_name = 'Historical Average' if i == 0 else ""

                ax.plot(mean_data['month'], mean_data['value'],
                        color='black', linewidth=3, marker='D', markersize=6,
                        zorder=5, label=label_name)  # zorder=5 forces the black line to stay on top

            # C) Format the panel
            ax.set_title(c_class.upper(), fontsize=14, fontweight='bold')
            ax.set_xticks(months)

            # Only add the month names if it's on the bottom row (to reduce clutter)
            if i >= 6:
                ax.set_xticklabels(month_names, rotation=45, fontsize=12)

            ax.set_ylabel('Monthly Use', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)

        # 6. Global Formatting and Legend
        # Attach a single master legend to the top-left plot but push it outside the data area
        axes[0].legend(title='Year', bbox_to_anchor=(1.02, 1), loc='upper left',
                       fontsize=10, title_fontsize=12, frameon=True)

        fig.suptitle('Monthly Water Use by Customer Sector', fontsize=20, y=0.98, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjusts the layout so the main title doesn't overlap

        plt.savefig('monthly_use_by_sector.png', dpi=300, bbox_inches='tight')
        plt.show()
        # plt.savefig('monthly_use_by_sector.png', dpi=300, bbox_inches='tight')
        # plt.close(fig)
        print("Saved multi-panel plot: monthly_use_by_sector.png")

    # --- Run the function ---
    plot_monthly_sector_use('/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/sector_monthly_use_long.csv')

def compare_withdrawal_diversions():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import calendar

    def compare_diversions_and_withdrawals(div_file, with_file):
        # --- 1. Process Diversions Data (Daily AF to Monthly/Annual kgal) ---
        div_df = pd.read_csv(div_file)
        div_df['Date'] = pd.to_datetime(div_df['Date'])
        val_col = div_df.columns[1]  # Automatically grabs the volume column

        # Convert Acre-Feet to kgal (1 AF = 325.851 kgal)
        div_df['value_kgal'] = div_df[val_col] * 325.851

        # Extract temporal groupings
        div_df['year'] = div_df['Date'].dt.year
        div_df['month'] = div_df['Date'].dt.month

        # Filter to requested 2019-2024 time domain
        div_df = div_df[(div_df['year'] >= 2019) & (div_df['year'] <= 2024)]

        # Calculate Monthly and Annual Sums
        div_monthly = div_df.groupby(['year', 'month'])['value_kgal'].sum().reset_index()
        div_monthly['Type'] = 'Diversions'

        div_annual = div_df.groupby('year')['value_kgal'].sum().reset_index()
        div_annual['Type'] = 'Diversions'

        # --- 2. Process Withdrawals Data (Summing Customer Classes) ---
        with_df = pd.read_csv(with_file)
        with_df.rename(columns={with_df.columns[0]: 'year'}, inplace=True)

        # Clean up column names by removing ' kgal' and whitespace
        with_df.columns = [col.replace('kgal', '').strip() for col in with_df.columns]

        # Filter to requested 2019-2024 time domain
        with_df = with_df[(with_df['year'] >= 2019) & (with_df['year'] <= 2024)]

        # Melt from wide to long format
        id_vars = ['year', 'Customer Category']
        value_vars = [col for col in with_df.columns if col not in id_vars]
        with_long = pd.melt(with_df, id_vars=id_vars, value_vars=value_vars, var_name='month_str', value_name='value')

        # Map string months to integers
        month_map = {name: i for i, name in enumerate(calendar.month_name) if name}
        with_long['month'] = with_long['month_str'].map(month_map)

        # Avoid Double Counting: 'TOTAL IRR (POTABLE + NP IRR)' is already represented by its individual components
        with_long = with_long[with_long['Customer Category'] != 'TOTAL IRR (POTABLE + NP IRR)']

        # Sum across all customer classes per month and year
        with_monthly = with_long.groupby(['year', 'month'])['value'].sum().reset_index()
        with_monthly.rename(columns={'value': 'value_kgal'}, inplace=True)
        with_monthly['Type'] = 'Withdrawals'

        # Calculate Annual Sums
        with_annual = with_monthly.groupby('year')['value_kgal'].sum().reset_index()
        with_annual['Type'] = 'Withdrawals'

        # --- 3. Format Data for Plotting ---
        # Merge the monthly datasets and map integers to 3-letter abbreviations
        combined_monthly = pd.concat([div_monthly, with_monthly], ignore_index=True)
        month_abbr = {i: calendar.month_abbr[i] for i in range(1, 13)}
        combined_monthly['month'] = combined_monthly['month'].map(month_abbr)

        # Convert month column to an ordered Categorical so it plots chronologically (Jan -> Dec)
        combined_monthly['month'] = pd.Categorical(combined_monthly['month'], categories=list(month_abbr.values()),
                                                   ordered=True)

        # Merge the annual datasets
        combined_annual = pd.concat([div_annual, with_annual], ignore_index=True)

        # --- 4. Plotting (Two Panels) ---
        # Create a figure with a 4:1 width ratio between the Monthly and Annual panels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [4, 1]})
        colors = ['#1f77b4', '#ff7f0e']  # Blue for Diversions, Orange for Withdrawals

        # Panel 1: Monthly Comparison
        sns.boxplot(data=combined_monthly, x='month', y='value_kgal', hue='Type', ax=ax1, palette=colors)
        ax1.set_title('Monthly Total Comparison (2019-2024)', fontsize=14)
        ax1.set_ylabel('Volume (kgal)', fontsize=12)
        ax1.set_xlabel('Month', fontsize=12)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
        ax1.legend(title='Record Type', loc='upper left')

        # Panel 2: Total Annual Comparison
        sns.boxplot(data=combined_annual, x='Type', y='value_kgal', hue='Type', ax=ax2, palette=colors, dodge=False)
        ax2.set_title('Total Annual Comparison', fontsize=14)
        ax2.set_ylabel('Volume (kgal)', fontsize=12)
        ax2.set_xlabel('Record Type', fontsize=12)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)

        # Hide the legend on the second panel since it's redundant
        ax2.legend().set_visible(False)

        # plt.suptitle('Diversions vs. Customer Withdrawals Comparison (2019-2024)', fontsize=18, fontweight='bold',
        #              y=1.02)

        plt.tight_layout()
        plt.show()
        # plt.savefig('diversions_vs_withdrawals.png', dpi=300, bbox_inches='tight')
        # plt.close()
        print("Saved comparison plot to: diversions_vs_withdrawals.png")

    # --- Run the function ---
    compare_diversions_and_withdrawals(
        div_file='/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/daily_diversions_2001-2024.csv',
        with_file='/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/sector_monthly_use_wide_2019-2025.csv'
    )

if __name__=='__main__':
    a=1
    extract_nc_data()
    merge_csv_files()
    weekly_resample_csvs()

    # plot_annualweekly_demand_timeseries(
    #                              rolling=False,
    #                              window_size=1)

    # var_autocorrelation()
    # convert_csvs()

    # calc_regression_weekly_models()

    # aggregate_demand()
    #
    # resample_daily_to_weekly()

    # multilinear_regression()

    # bootstrap_mlr()

    # plot_monthly_customerclass()

    # compare_withdrawal_diversions()

    bootstrap_MLR_offsets()