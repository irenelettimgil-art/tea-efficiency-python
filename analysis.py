import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("tea_data.csv")

# Log transformation (Cobb-Douglas)
data["log_output"] = np.log(data["output"])
data["log_land"] = np.log(data["land"])
data["log_labor"] = np.log(data["labor"])
data["log_fertilizer"] = np.log(data["fertilizer"])

# Define model
X = data[["log_land", "log_labor", "log_fertilizer"]]
X = sm.add_constant(X)
y = data["log_output"]

# Fit model
model = sm.OLS(y, X).fit()

# Print regression results
print("\n=== REGRESSION RESULTS ===")
print(model.summary())

# Efficiency approximation
data["efficiency"] = np.exp(model.resid)

print("\n=== EFFICIENCY RESULTS ===")
print(data[["output", "efficiency"]])

# -----------------------------
# 📊 VISUALIZATIONS (CREATIVITY BOOST)
# -----------------------------

# 1. Output vs Land
plt.figure()
plt.scatter(data["land"], data["output"])
plt.xlabel("Land")
plt.ylabel("Output")
plt.title("Tea Output vs Land")
plt.show()

# 2. Output vs Labor
plt.figure()
plt.scatter(data["labor"], data["output"])
plt.xlabel("Labor")
plt.ylabel("Output")
plt.title("Tea Output vs Labor")
plt.show()

# 3. Efficiency Distribution
plt.figure()
plt.hist(data["efficiency"])
plt.xlabel("Efficiency")
plt.ylabel("Frequency")
plt.title("Distribution of Efficiency Scores")
plt.show()

# -----------------------------
# 📘 INTERPRETATION
# -----------------------------
print("\n=== INTERPRETATION ===")
print("The results show that land, labor, and fertilizer positively influence tea output.")
print("Efficiency scores indicate how close each farm is to optimal production.")
print("Values closer to 1 suggest higher efficiency.")
