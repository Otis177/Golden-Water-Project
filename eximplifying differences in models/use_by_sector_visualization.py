import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model  import LinearRegression
from sklearn.linear_model import Ridge

with2019 = False

#Sets the output files for the graphs
if (with2019 == True):
    outputPaths = ["C:\\Users\\eo\\Downloads\\Golden project\\eximplifying differences in models\\Monthly_Use_Percents.png","C:\\Users\\eo\\Downloads\\Golden project\\eximplifying differences in models\\Monthly_Use_Raws_WO2019.png"]
else:
    outputPaths = ["C:\\Users\\eo\\Downloads\\Golden project\\eximplifying differences in models\\Monthly_Use_Percents_WO2019.png","C:\\Users\\eo\\Downloads\\Golden project\\eximplifying differences in models\\Monthly_Use_Raws.png"]
#Geta the file and makes it a dataframe  
df = pd.read_csv("C:\\Users\\eo\\Downloads\\Golden project\\data\\sector_monthly_use_long.csv")
#Removes all 2019 data from the dataframe since it is dramaically different than the other years and would skew the data
if (with2019 == False):
    df = df[df['year'] != 2019]
#Makes a cleaned dataframe without any aggrigated totals so I can easily group by month and year to get the total usage for each month
dfClean = df[~df['customer class'].isin(['TOTAL IRES TIERS','TOTAL IRR (POTABLE + NP IRR)'])]
#Makes another new dataframe with the leftover "dirty" data not included in the total
dfDirty = df[df['customer class'].isin(['TOTAL IRES TIERS','TOTAL IRR (POTABLE + NP IRR)'])]
#Creates a total value for each month and year by grouping the cleaned dataframe by year and month and summing the usage values
dfMonthlyTotal = dfClean.groupby(['year', 'month']).sum().reset_index()
#Renames the customer class data for each month year pair to be "TOTAL" instead of "ICOMMIMULTNP IRROCOMMORESPOTABLE IRRTOTAL IRES..."
dfMonthlyTotal['customer class'] = 'TOTAL'
#Combines all the dataframes
df = pd.concat([dfClean, dfMonthlyTotal], ignore_index=True)
df = pd.concat([df, dfDirty], ignore_index=True)
#sorts the dataframe by month and then customer class to make it easier to find the usage by class on any given month later
df =  df.sort_values(by=['month', 'customer class']).reset_index(drop=True)
#combines the value data for each month and spesific sector into a single piece of data for each spesific sector than averages it
# OTIS DO THIS IF YOU WANT ROUNDED dfAveraged = df.groupby(['month', 'customer class']).mean().round().reset_index()
dfAveraged = df.groupby(['month', 'customer class']).mean().reset_index()
#gets rid of year as it is unneeded
dfAveraged = dfAveraged.drop(columns=['year'])
#adds a new column to the averaged dataframe that is the percent of total usage each customer class uses for each month
totals = dfAveraged[dfAveraged['customer class'] == 'TOTAL'] \
    .set_index('month')['value']

dfAveraged['percent_of_total'] = (
    dfAveraged['value'] / dfAveraged['month'].map(totals) * 100
)
print(dfAveraged['customer class'].unique())
print(dfAveraged.shape)
print(dfAveraged.head(50)) 

#---------------------------MAKE PERCENTAGE PLOT-----------------------------
fig, ax = plt.subplots(figsize=(12, 9))
dfForPercents = dfAveraged[~dfAveraged['customer class'].isin(['TOTAL IRES TIERS','TOTAL IRR (POTABLE + NP IRR)','TOTAL'])]
pivot = dfForPercents.pivot(
    index='month',
    columns='customer class',
    values='percent_of_total'
).fillna(0)

bottom = np.zeros(len(pivot))

for col in pivot.columns:
    ax.bar(
        pivot.index,
        pivot[col],
        bottom=bottom,
        label=col
    )
    bottom += pivot[col].values
if (with2019 == True):
    ax.set_title("Monthly Water Usage Share by Customer Class")
else:
    ax.set_title("Monthly Water Usage Share by Customer Class Without 2019")
ax.set_xlabel("Month")
ax.set_ylabel("Percent of Total Usage (%)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(outputPaths[0], dpi=300, bbox_inches='tight')
#---------------------------MAKE RAW DATA PLOT-----------------------------
fig2, ax2 = plt.subplots(figsize=(12, 9))
df_lines = dfAveraged.copy()

for cls in df_lines['customer class'].unique():
    subset = df_lines[df_lines['customer class'] == cls]
    ax2.plot(
        subset['month'],
        subset['value'],
        marker='o',
        label=cls
    )
if (with2019 == True):
    ax2.set_title("Monthly Water Usage by Customer Class (Raw Values)")
else:
    ax2.set_title("Monthly Water Usage by Customer Class Without 2019 (Raw Values)")
ax2.set_xlabel("Month")
ax2.set_ylabel("Usage")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(outputPaths[1], dpi=300, bbox_inches='tight')