import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ridgeData = {
    "Output":"c:\\Users\\eo\\Downloads\\Golden project\\Find best alpha\\Ridge_Regression",
    "Type": "Ridge",
    0.25:{"Alpha":0.25, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_0.25_r2_100_iterations.csv"},
    0.5:{"Alpha":0.5, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_0.5_r2_100_iterations.csv"},
    1.0:{"Alpha":1.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_1.0_r2_100_iterations.csv"},
    1.5:{"Alpha":1.5, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_1.5_r2_100_iterations.csv"},
    2.0:{"Alpha":2.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_2.0_r2_100_iterations.csv"},
    3.0:{"Alpha":3.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_3.0_r2_100_iterations.csv"},
    5.0:{"Alpha":5.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_5.0_r2_100_iterations.csv"},
    10.0:{"Alpha":10.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_10.0_r2_100_iterations.csv"},
    25.0:{"Alpha":25.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_25.0_r2_100_iterations.csv"},
    50.0:{"Alpha":50.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Ridge\\Data\\r2\\Ridge_alpha_50.0_r2_100_iterations.csv"}
}


lassoData = {
    "Output":"c:\\Users\\eo\\Downloads\\Golden project\\Find best alpha\\Lasso_Regression",
    "Type": "Lasso",
    0.25:{"Alpha":0.25, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_0.25_r2_100_iterations.csv"},
    0.5:{"Alpha":0.5, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_0.5_r2_100_iterations.csv"},
    1.0:{"Alpha":1.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_1.0_r2_100_iterations.csv"},
    1.5:{"Alpha":1.5, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_1.5_r2_100_iterations.csv"},
    2.0:{"Alpha":2.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_2.0_r2_100_iterations.csv"},
    3.0:{"Alpha":3.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_3.0_r2_100_iterations.csv"},
    5.0:{"Alpha":5.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_5.0_r2_100_iterations.csv"},
    10.0:{"Alpha":10.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_10.0_r2_100_iterations.csv"},
    25.0:{"Alpha":25.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_25.0_r2_100_iterations.csv"},
    50.0:{"Alpha":50.0, "Path":"c:\\Users\\eo\\Downloads\\Golden project\\Output data\\Lasso\\Data\\r2\\Lasso_alpha_50.0_r2_100_iterations.csv"}
}


def find_best_alpha_values(data):
    """Loads all alpha files and creates comparison graphs"""

    os.makedirs(data["Output"], exist_ok=True)

    justPaths = [
        val for key, val in data.items()
        if isinstance(key, (int, float))
    ]

    df = pd.DataFrame()

    for path in justPaths:
        alpha = path["Alpha"]
        tempFilePath = path["Path"]

        tempDF = pd.read_csv(tempFilePath)

        df[alpha] = tempDF["test_r2"]

    df["Best Alpha"] = df.drop(columns=["Best Alpha"], errors="ignore").idxmax(axis=1)

    print(df)

    best_alpha_line_graph(
        df,
        data["Output"] + "\\Best_Alpha_Line_Graph.png",
        data["Type"]
    )

    alpha_heatmap(
        df,
        data["Output"] + "\\Alpha_Heatmap.png",
        data["Type"]
    )

    alpha_frequency(
        df,
        data["Output"] + "\\Alpha_Frequency.png",
        data["Type"]
    )

    best_alpha_weekly(
        df,
        data["Output"] + "\\Best_Alpha_Weekly.png",
        data["Type"]
    )

    average_alpha_graph(
        df,
        data["Output"] + "\\Average_Alpha_Graph.png",
        data["Type"]
    )

    save_best_alpha_values(
        df,
        data["Output"] + "\\Best_Alpha_Values.csv"
    )


def best_alpha_line_graph(dataFrame, outputPath, modelType):
    """Line graph of all alpha values"""

    alpha_columns = [
        col for col in dataFrame.columns
        if col != "Best Alpha"
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    for alpha in alpha_columns:
        ax.plot(
            dataFrame.index + 1,
            dataFrame[alpha],
            label=f"Alpha {alpha}",
            linewidth=2
        )

    ax.set_xlabel("Week")
    ax.set_ylabel("Test $R^2$")
    ax.set_title(f"Weekly Test $R^2$ by {modelType} Alpha")

    ax.set_xticks(range(1,53,4))
    ax.set_xlim(1,52)
    ax.margins(x=0)

    ax.grid(alpha=.3)
    ax.legend(title="Alpha", ncol=2)

    plt.tight_layout()

    fig.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Line graph saved...")


def alpha_heatmap(dataFrame, outputPath, modelType):

    alpha_columns = [
        col for col in dataFrame.columns
        if col != "Best Alpha"
    ]

    fig, ax = plt.subplots(figsize=(12,5))

    image = ax.imshow(
        dataFrame[alpha_columns].T,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Alpha")

    ax.set_xticks(range(0,52,4))
    ax.set_xticklabels(range(1,53,4))

    ax.set_yticks(range(len(alpha_columns)))
    ax.set_yticklabels(alpha_columns)

    ax.set_title(f"{modelType} Alpha Test $R^2$ Heatmap")

    fig.colorbar(image, label="Test $R^2$")

    plt.tight_layout()

    fig.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Heatmap saved...")


def alpha_frequency(dataFrame, outputPath, modelType):

    counts = dataFrame["Best Alpha"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Number of Weeks")
    ax.set_title(f"Number of Weeks Each {modelType} Alpha Was Best")

    plt.tight_layout()

    fig.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Frequency graph saved...")


def best_alpha_weekly(dataFrame, outputPath, modelType):

    fig, ax = plt.subplots(figsize=(12,4))

    ax.step(
        dataFrame.index + 1,
        dataFrame["Best Alpha"],
        where="mid",
        linewidth=2
    )

    ax.scatter(
        dataFrame.index + 1,
        dataFrame["Best Alpha"],
        s=35
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Best Alpha")
    ax.set_title(f"Best {modelType} Alpha By Week")

    ax.set_xticks(range(1,53,4))

    ax.grid(alpha=.3)

    plt.tight_layout()

    fig.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Best alpha graph saved...")

def average_alpha_graph(df, outputPath, modelType):

    alpha_columns = [
        col for col in df.columns
        if col != "Best Alpha"
    ]

    averages = df[alpha_columns].mean()

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(
        averages.index,
        averages.values,
        marker="o"
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Average Test R²")
    ax.set_title(f"Average Test R² by {modelType} Alpha")

    ax.set_xscale("log")
    ax.grid(alpha=0.3)

    fig.savefig(outputPath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("average alpha graph saved...")

def save_best_alpha_values(df, outputPath):
    """Saves the best alpha values to a CSV file"""

    best_alpha = df["Best Alpha"]
    best_alpha.to_csv(outputPath, header=["Alpha"], index_label="Week")

    print(f"Best alpha values saved to {outputPath}...")


find_best_alpha_values(ridgeData)
find_best_alpha_values(lassoData)