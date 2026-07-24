import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
1. get file names
2. Open files into pandas dataframes
3. cut down to just the weeks with the most water usage weeks 20-40
4. make the x axis of the graph be titles year type with 4 different year types (wet dry normal and random) and the y axis be the r2 values for each model
5. for each of the models add their respective r2 values. do so for each year typeeach year type will have 3 different color coded boxes asccoetated with it representing each model type."""
def plot_year_type_comparison(outputPath, trainedAndTested = False):
    # Define the file paths for the CSV files
    file_paths_tested_only = {
        "RidgeDry": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_random_tested_on_Dry.csv",
        "RidgeWet": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_random_tested_on_Wet.csv",
        "RidgeNormal":"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_random_tested_on_Normal.csv",
        "RidgeRandom": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_random_tested_on_Random.csv",
        "LassoDry": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_random_tested_on_Dry.csv",
        "LassoWet": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_random_tested_on_Wet.csv",
        "LassoNormal": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_random_tested_on_Normal.csv",
        "LassoRandom": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_random_tested_on_Random.csv",
        "MLRDry": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_random_tested_on_Dry.csv",
        "MLRWet": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_random_tested_on_Wet.csv",
        "MLRNormal": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_random_tested_on_Normal.csv",
        "MLRRandom": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_random_tested_on_Random.csv"}
    file_paths_trained_and_tested = {
        "RidgeDry": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_Dry_tested_on_Dry.csv", 
        "RidgeWet": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_Wet_tested_on_Wet.csv",
        "RidgeNormal": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_Normal_tested_on_Normal.csv",
        "RidgeRandom": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestRidge\\Data\\r2\\BestRidge_best_alpha_r2_100_iterations_trained_on_random_tested_on_Random.csv",
        "LassoDry": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_Dry_tested_on_Dry.csv",
        "LassoWet": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_Wet_tested_on_Wet.csv",
        "LassoNormal": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_Normal_tested_on_Normal.csv",
        "LassoRandom": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\BestLasso\\Data\\r2\\BestLasso_best_alpha_r2_100_iterations_trained_on_random_tested_on_Random.csv",
        "MLRDry": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_Dry_tested_on_Dry.csv",
        "MLRWet": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_Wet_tested_on_Wet.csv",
        "MLRNormal": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_Normal_tested_on_Normal.csv",
        "MLRRandom": "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\MLR\\Data\\R2\\Multiple_Linear_Regression_r2_100_iterations_trained_on_random_tested_on_Random.csv"}
    minWeek = 27
    maxWeek = 34
    # Load all files and filter to peak demand weeks (27-34)
    dryDF = pd.DataFrame()
    filesToRead = file_paths_tested_only
    if trainedAndTested == False:
        filesToRead = file_paths_tested_only
    else:
        filesToRead = file_paths_trained_and_tested
    dryDF['week'] = range(1, 53)
    dryDF['Ridge'] = pd.read_csv(filesToRead['RidgeDry'])['test_r2'].values
    dryDF['Lasso'] = pd.read_csv(filesToRead['LassoDry'])['test_r2'].values
    dryDF['MLR'] = pd.read_csv(filesToRead['MLRDry'])['test_r2'].values
    wetDF = pd.DataFrame()
    wetDF['week'] = range(1, 53)
    wetDF['Ridge'] = pd.read_csv(filesToRead['RidgeWet'])['test_r2'].values
    wetDF['Lasso'] = pd.read_csv(filesToRead['LassoWet'])['test_r2'].values
    wetDF['MLR'] = pd.read_csv(filesToRead['MLRWet'])['test_r2'].values
    normalDF = pd.DataFrame()
    normalDF['week'] = range(1, 53)
    normalDF['Ridge'] = pd.read_csv(filesToRead['RidgeNormal'])['test_r2'].values
    normalDF['Lasso'] = pd.read_csv(filesToRead['LassoNormal'])['test_r2'].values
    normalDF['MLR'] = pd.read_csv(filesToRead['MLRNormal'])['test_r2'].values
    randomDF = pd.DataFrame()
    randomDF['week'] = range(1, 53)
    randomDF['Ridge'] = pd.read_csv(filesToRead['RidgeRandom'])['test_r2'].values
    randomDF['Lasso'] = pd.read_csv(filesToRead['LassoRandom'])['test_r2'].values
    randomDF['MLR'] = pd.read_csv(filesToRead['MLRRandom'])['test_r2'].values
    
    #Cuts the DFs down to weeks 20-40
    dryDF = dryDF[(dryDF['week'] >= minWeek) & (dryDF['week'] <= maxWeek)]
    wetDF = wetDF[(wetDF['week'] >= minWeek) & (wetDF['week'] <= maxWeek)]
    normalDF = normalDF[(normalDF['week'] >= minWeek) & (normalDF['week'] <= maxWeek)]
    randomDF = randomDF[(randomDF['week'] >= minWeek) & (randomDF['week'] <= maxWeek)]

    #gets a dict of the mean r2 values for each model for each year type
    r2Values = {
        'Dry': {
            'Ridge': dryDF['Ridge'].mean(),
            'Lasso': dryDF['Lasso'].mean(),
            'MLR': dryDF['MLR'].mean()
        },
        'Wet': {
            'Ridge': wetDF['Ridge'].mean(),
            'Lasso': wetDF['Lasso'].mean(),
            'MLR': wetDF['MLR'].mean()
        },
        'Normal': {
            'Ridge': normalDF['Ridge'].mean(),
            'Lasso': normalDF['Lasso'].mean(),
            'MLR': normalDF['MLR'].mean()
        },
        'Random': {
            'Ridge': randomDF['Ridge'].mean(),
            'Lasso': randomDF['Lasso'].mean(),
            'MLR': randomDF['MLR'].mean()
        }
    }

    yearTypes = ['Dry', 'Wet', 'Normal', 'Random']
    models = ['Ridge', 'Lasso', 'MLR']
    colorMap = {
        'Ridge': '#4878CF',
        'Lasso': '#6BAF45',
        'MLR':   '#D65F5F'
    }

    #Create Boxplot
    nYearTypes = len(yearTypes)  # 4
    nModels = len(models)        # 3
    width = 0.25
    x = np.arange(nYearTypes)   # positions for each year type group

    fig, ax = plt.subplots(figsize=(5.7, 4.25))

    for i, model in enumerate(models):
        values = [r2Values[yt][model] for yt in yearTypes]
        ax.bar(x + i * width, values, width, label=model, color=colorMap[model])
    ax.set_xticks(x + width)  # offset by one width to center under the middle bar
    ax.set_xticklabels(yearTypes)
    ax.set_ylabel(r'$R^2$ Score')
    if trainedAndTested:
        ax.set_title(f'Model Skill Comparison by Year Type\nTrained and Tested on Specified Year Type Weeks ({minWeek}-{maxWeek})')
    else:
        ax.set_title(f'Model Skill Comparison by Year Type\nTrained on all Year Types Tested on Specified Year Type Weeks ({minWeek}-{maxWeek})')
    ax.legend(title='Model')
    ax.set_ylim(-0.1, 1.0)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')

outputPath = "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\Year_Type_Comparison_Trained_On_Random_Tested_On_Type.png"
plot_year_type_comparison(outputPath, trainedAndTested= False)
outputPath = "C:\\Users\\eo\\Downloads\\Golden project\\Output data\\Year_Type_Comparison_Trained_On_Type_Tested_On_Type.png"
plot_year_type_comparison(outputPath, trainedAndTested= True)