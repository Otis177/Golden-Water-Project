import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_data(df, modelName, nIterations, alpha = 1.0,r2CSVOutputPath = None, yearType = None):
    weeks = 52
    pathModelName = modelName
    if modelName == "Ridge" or modelName == "Lasso":
        pathModelName += f"_alpha_{alpha}"
    if pathModelName == "Ridge_alpha_0.0":
        pathModelName = "Multiple_Linear_Regression"
    outputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelName}\\Graphs\\{pathModelName}_skill_boxplot_{nIterations}_iterations.png"
    if modelName[:4] == "Best":
        pathModelName += f"_best_alpha"
        outputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelName[4:]}\\Graphs\\{pathModelName}_skill_boxplot_{nIterations}_iterations.png"
    if yearType != "None":
        outputPath = f"C:\\Users\\eo\\Downloads\\Golden project\\Output data\\{modelName}\\Graphs\\{pathModelName}_skill_boxplot_{nIterations}_iterations_trained_on_random_tested_on_{yearType}.png"
    modelNameConverter = {
        "Ridge": "Ridge Regression",
        "Lasso": "Lasso Regression",
        "BestRidge": "Best Ridge Regression",
        "BestLasso": "Best Lasso Regression",
        "MLR": "Multiple Linear Regression"}
    trainData = df.groupby('week')['train_r2'].apply(list).sort_index()
    testMedians = df.groupby('week')['test_r2'].median().sort_index()
    print("Data grouped")
    print("Saving r2 results to CSV...")
    testMedians.to_csv(r2CSVOutputPath, index=False)
    print(f"Results saved to {r2CSVOutputPath}")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)
    bp = ax.boxplot(trainData,
                    positions = np.arange(1, weeks + 1),
                    patch_artist = True,
                    notch = True,
                    showfliers=False)
    for box in bp['boxes']:
        box.set(facecolor='#6BAF45', alpha=1.0)
    print("Boxplot created")
    print(len(np.arange(1, weeks + 1)))
    print(len(testMedians))
    print(np.arange(1, weeks + 1))
    print(testMedians)
    ax.scatter(
                np.arange(1, weeks + 1),
                testMedians,
                marker='*',
                s=60,
                color='black',
                label='Median Out-of-Sample Skill')
    print("Scatter plot created")
    ax.axhline(
                y=0,
                color='red',
                linestyle='--',
                linewidth=1,
                label='Zero Skill Baseline')
    print("Zero skill line created")
    ax.set_xlabel('Week of the Year')
    ax.set_ylabel(r'$R^2$ Score')
    trueModelName = modelNameConverter[modelName]
    if modelName == "Ridge" or modelName == "Lasso":
        trueModelName += f" (alpha={alpha})"
    if modelName == "Ridge" and alpha == 0.0:
        trueModelName = "Multiple Linear Regression"
    ax.set_title(
    f'{trueModelName} w/Antecedent Lags ({nIterations} Iterations) \n'
    'Boxes = In-Sample Training $R^2$ | '
    'Stars = Median Out-of-Sample Testing $R^2$')
    ax.set_xticks(np.arange(1, weeks+1))
    ax.set_ylim(-0.25, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')

def plot_all_data(listOfModelResults, nIterations, outputPath = None):
    """
    Plots a combined skill boxplot for all models.

    Parameters:
        listOfModelResults: List of tuples (df, modelName)
        nIterations: Number of bootstrap iterations
        outputPath: Path to save the plot
    """
    width = 0.25  # width of each box
    colorMap = {
        "Ridge":      "#4878CF",  # blue
        "BestRidge":  "#4878CF",  # blue (same family)
        "Lasso":      "#6BAF45",  # green
        "BestLasso":  "#6BAF45",  # green (same family)
        "MLR":        "#D65F5F",  # red
    }

    nModels = len(listOfModelResults)
    # Width of each box and total spread across all models per week
    width = 0.7 / nModels
    
    # Center offsets so boxes are centered around each week tick
    offsets = np.linspace(-(nModels - 1) / 2 * width, 
                           (nModels - 1) / 2 * width, 
                           nModels)
    weeks = np.arange(1, 53)
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)
    for idx, (modelName, df) in enumerate(listOfModelResults):
        color = colorMap.get(modelName, "#000000")  # default to black if not found
        positions = weeks + offsets[idx]
        trainData = df.groupby('week')['train_r2'].apply(list).sort_index()
        testMedians = df.groupby('week')['test_r2'].median().sort_index()
        #Draw Boxes
        bp = ax.boxplot(trainData,
                        positions=positions,
                        widths=width * 0.9,
                        patch_artist=True,
                        notch=False,  # notch=True gets messy with small offsets
                        showfliers=False,
                        manage_ticks=False)
        
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.7)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set(color=color)
        
        # Draw stars for out-of-sample median
        ax.scatter(positions, testMedians,
                   marker='*', s=60,
                   color=color, zorder=5,
                   label=modelName)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Skill Baseline')
    ax.set_xlim(0.5, 52.5)
    ax.set_xticks(weeks)
    ax.set_ylim(-0.25, 1.0)
    ax.set_xlabel('Week of the Year')
    ax.set_ylabel(r'$R^2$ Score')
    ax.set_title(f'Model Skill Comparison w/ Antecedent Lags ({nIterations} Iterations)\n'
                 'Boxes = In-Sample Training $R^2$ | Stars = Median Out-of-Sample Testing $R^2$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')