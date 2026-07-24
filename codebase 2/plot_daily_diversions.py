import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar

def plot_daily_diversions(filePath, outputPath):
    # Load and parse the data
    df = pd.read_csv(filePath)
    df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y')
    df['dayofyear'] = df['Date'].dt.dayofyear
    df.rename(columns={'Total Diverted Raw + Seasonal AF': 'usage'}, inplace=True)

    # Average across all years for each day of year
    dailyAvg = df.groupby('dayofyear')['usage'].mean()

    # Plot
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(dailyAvg.index, dailyAvg.values, color='#4878CF', linewidth=1.5)
    ax.fill_between(dailyAvg.index, dailyAvg.values, alpha=0.3, color='#4878CF')

    ax.set_xlabel('Month')
    ax.set_ylabel('Acre-Feet')
    ax.set_title('Average Daily Water Diversions — City of Golden, CO (2001-2024)')
    ax.set_xlim(1, 365)
    monthStarts = [pd.Timestamp(f'2023-{m:02d}-01').dayofyear for m in range(1, 13)]
    monthLabels = [calendar.month_abbr[m] for m in range(1, 13)]
    ax.set_xticks(monthStarts)
    ax.set_xticklabels(monthLabels)
    ax.set_ylim(0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {outputPath}")

plot_daily_diversions(
    'C:\\Users\\eo\\Downloads\\Golden project\\data\\daily_diversions_2001-2024.csv',
    'C:\\Users\\eo\\Downloads\\Golden project\\Output data\\daily_diversions_avg.png'
)