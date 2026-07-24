import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model  import LinearRegression
from sklearn.linear_model import Ridge
data = {
    "Precipitation": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Usage":         [1, 5, 4, 11, 6, 11, 9, 11, 11, 15]
}
df = pd.DataFrame(data)
precipReshaped = df["Precipitation"].values.reshape(-1, 1)
xTraining = precipReshaped[0:4]
yTraining = df["Usage"][0:4]
print (df)
print(xTraining)
Ridge = Ridge(alpha=5.0)
linearRegression = LinearRegression()
Ridge.fit(xTraining, yTraining)
linearRegression.fit(xTraining, yTraining)
xLine = np.linspace(0, 10, 500).reshape(-1, 1)
yLinear = linearRegression.predict(xLine)
yRidge = Ridge.predict(xLine)
print(yLinear)
fig, ax = plt.subplots(figsize=(9, 10))
ax.scatter(data["Precipitation"][0:4], yTraining, color='red', marker='o', s=400, label='Training Data')
ax.scatter(data["Precipitation"][4:], df["Usage"][4:], color='green', marker='o', s=400, label='Test Data')
ax.plot(xLine, yLinear, color='blue', label='Linear Regression')
ax.plot(xLine, yRidge, color='orange', label='Ridge Regression')
ax.set_title("Ridge vs Linear Regression")
ax.set_xlabel("Precipitation")
ax.set_ylabel("Usage")
ax.legend()
ax.grid(True)
outputPath = "C:\\Users\\eo\\Downloads\\Golden project\\eximplifying differences in models\\Ridge_VS_Linear_Regression.png"
plt.tight_layout()
plt.savefig(outputPath, dpi=300, bbox_inches='tight')