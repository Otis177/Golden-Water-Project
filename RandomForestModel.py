import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import calendar


def run_bootstrap_random_forest(diversion_file, variable_files, pop_data,
                                offsets=[0, -1, -2], window=5, n_iterations=1000, n_train=17):
    # 1. Load the Target Dataset (Weekly Diversions -> GPC)
    df_target = pd.read_csv(diversion_file)
    df_target.rename(columns={'value': 'target_af'}, inplace=True)
    df_target['total_gallons'] = df_target['target_af'] * 325851
    df_target['population'] = df_target['year'].map(pop_data)
    df_target['target'] = df_target['total_gallons'] / df_target['population']

    # 2. Load, Generate Lags, and Merge Predictor Variables
    df = df_target.dropna(subset=['target']).copy()
    feature_names = []

    for var_name, var_file in variable_files.items():
        print(f"Loading {var_name.upper()} and generating lags...")
        df_var = pd.read_csv(var_file)
        grid_cols = [c for c in df_var.columns if c not in ['year', 'week']]
        df_var['regional_mean'] = df_var[grid_cols].mean(axis=1)
        df_var.sort_values(['year', 'week'], inplace=True)

        cols_to_merge = ['year', 'week']

        for offset in offsets:
            lag_num = abs(offset)
            col_name = f"{var_name}_lag{lag_num}"
            df_var[col_name] = df_var['regional_mean'].shift(-offset)

            cols_to_merge.append(col_name)
            feature_names.append(col_name)

        df = pd.merge(df, df_var[cols_to_merge], on=['year', 'week'], how='inner')

    df.dropna(subset=feature_names, inplace=True)
    all_years = df['year'].unique()
    max_week = int(df['week'].max())

    # Pre-extract rolling window data
    window_data_map = {}
    for week in range(1, max_week + 1):
        window_weeks = [(w - 1) % max_week + 1 for w in range(int(week) - window, int(week) + window + 1)]
        window_data_map[week] = df[df['week'].isin(window_weeks)].copy()

    # 3. Bootstrap Resampling Loop
    print(f"Starting {n_iterations} Random Forest bootstrap iterations...")
    results = []
    importance_records = []

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

            X_train = train_df[feature_names].values
            y_train = train_df['target'].values

            X_test = test_df[feature_names].values
            y_test = test_df['target'].values

            # --- RANDOM FOREST MODEL ---
            # Max_depth=3 prevents the model from perfectly memorizing the tiny 17-year training set
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=None,
                                             n_jobs=-1)
            rf_model.fit(X_train, y_train)

            # Testing
            y_pred = rf_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            train_r2 = rf_model.score(X_train, y_train)

            results.append({
                'iteration': i,
                'week': week,
                'train_r2': train_r2,
                'test_r2': test_r2
            })

            # Record Feature Importances
            importances = rf_model.feature_importances_
            imp_dict = {'iteration': i, 'week': week}
            for j, f_name in enumerate(feature_names):
                imp_dict[f_name] = importances[j]
            importance_records.append(imp_dict)

    results_df = pd.DataFrame(results)
    imp_df = pd.DataFrame(importance_records)

    # --- 4. Process Data for Plotting ---
    box_data = []
    median_test_skills = []

    for w in range(1, max_week + 1):
        w_data = results_df[results_df['week'] == w]
        box_data.append(w_data['train_r2'].values)

        test_r2_vals = w_data['test_r2'].values
        test_r2_vals = test_r2_vals[test_r2_vals > -2.0]

        if len(test_r2_vals) > 0:
            median_test_skills.append(np.median(test_r2_vals))
        else:
            median_test_skills.append(np.nan)

    # --- 5. Generate Skill Boxplot ---
    fig1, ax1 = plt.subplots(figsize=(20, 8))

    bp = ax1.boxplot(box_data, positions=range(1, max_week + 1),
                     patch_artist=True, showfliers=False, widths=0.7)

    for box in bp['boxes']:
        box.set_facecolor('#2ca02c')  # Random Forest Green
        box.set_edgecolor('black')
        box.set_alpha(0.7)

    ax1.plot(range(1, max_week + 1), median_test_skills,
             marker='*', color='black', linestyle='None',
             markersize=10, zorder=5, label='Median Out-of-Sample Skill')

    ax1.axhline(0, color='red', linestyle='--', alpha=0.7, zorder=0, label='Zero Skill Baseline')
    ax1.set_xlabel('Week of the Year', fontsize=12, labelpad=30)
    ax1.set_ylabel('$R^2$ Score', fontsize=12)
    ax1.set_title(
        f'Random Forest Model Skill w/ Antecedent Lags ({n_iterations} Iterations)\nMax Depth = 3 | Min Samples/Leaf = 2\nBoxes = In-Sample Training $R^2$ | Stars = Median Out-of-Sample Testing $R^2$',
        fontsize=15)

    # Map Weeks to Months (Secondary X-Axis)
    weeks_array = np.arange(1, max_week + 1)
    dummy_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta((weeks_array - 1) * 7, unit='D')
    month_series = dummy_dates.month

    month_centers, month_labels = [], []
    for m in range(1, 13):
        weeks_in_m = weeks_array[month_series == m]
        if len(weeks_in_m) > 0:
            month_centers.append(np.mean(weeks_in_m))
            month_labels.append(calendar.month_abbr[m])
            if m > 1:
                ax1.axvline(weeks_in_m[0] - 0.5, color='gray', linestyle=':', alpha=0.4, zorder=0)

    ax_months = ax1.secondary_xaxis('bottom')
    ax_months.spines['bottom'].set_position(('outward', 30))
    ax_months.set_xticks(month_centers)
    ax_months.set_xticklabels(month_labels, fontsize=12, fontweight='bold')
    ax_months.tick_params(axis='x', length=0)

    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_xticks(range(1, 53))
    plt.ylim(-1.0, 1.2)

    fig1.tight_layout()
    # plt.savefig('rf_bootstrap_skill.png', dpi=300, bbox_inches='tight')
    # plt.close(fig1)
    print("Saved RF Skill Plot: rf_bootstrap_skill.png")

    # --- 6. Generate Feature Importance Plot ---
    # Group by week and average the feature importances across all 1000 iterations
    avg_importances = imp_df.groupby('week')[feature_names].mean()

    fig2, ax2 = plt.subplots(figsize=(20, 8))

    # Create a stacked bar chart of the importances
    avg_importances.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20', width=0.85, edgecolor='black',
                         linewidth=0.5)

    ax2.set_title(f'Random Forest Average Feature Importance by Week\n(What variables is the model relying on?)',
                  fontsize=15)
    ax2.set_xlabel('Week of the Year', fontsize=12, labelpad=30)
    ax2.set_ylabel('Relative Importance (Adds to 1.0)', fontsize=12)

    # Month demarcation lines for the second plot
    for m in range(2, 13):
        weeks_in_m = weeks_array[month_series == m]
        if len(weeks_in_m) > 0:
            ax2.axvline(weeks_in_m[0] - 1.5, color='black', linestyle='--', alpha=0.8,
                        zorder=5)  # -1.5 adjusts for zero-indexed categorical bars

    ax_months2 = ax2.secondary_xaxis('bottom')
    ax_months2.spines['bottom'].set_position(('outward', 30))

    # Shift month centers slightly for categorical plotting
    cat_month_centers = [c - 1 for c in month_centers]
    ax_months2.set_xticks(cat_month_centers)
    ax_months2.set_xticklabels(month_labels, fontsize=12, fontweight='bold')
    ax_months2.tick_params(axis='x', length=0)

    ax2.legend(title='Variables & Lags', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=0)

    fig2.tight_layout()
    plt.show()
    # plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    # plt.close(fig2)
    print("Saved RF Feature Importance Plot: rf_feature_importance.png\n")


# --- Configuration Section ---
anomaly_window = 5

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
    'vpd': '/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/vpd_weekly_resampled.csv',
    'tmmx':'/Volumes/LivnehExt01/Gridmet/golden_tabular/merged/weekly/tmmx_weekly_resampled.csv'
}

# The RF will use ALL of these lags simultaneously
lag_offsets = [0, -1, -2]

run_bootstrap_random_forest(
    diversion_file='/Users/nebj6909/Library/CloudStorage/OneDrive-UCB-O365/Documents/City of Golden/Data/weekly_diversions.csv',
    variable_files=predictor_variables,
    pop_data=population_table,
    offsets=lag_offsets,
    window=anomaly_window,
    n_iterations=25,
    n_train=17
)