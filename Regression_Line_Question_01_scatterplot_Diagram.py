import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Given data points
x = np.array([3, 1, 3, 5]) # Independent variable
y = np.array([5, 8, 6, 4]) # Dependent variable

# Calculate the slope and intercept using linregress
slope, intercept, r_value, p_value, std_err =
linregress(x, y)

# Generate predicted y values based on the regression line
y_pred = intercept + slope * x

# Create scatter plot of the original data points
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label="Data Points")

# Plot the regression line
plt.plot(x, y_pred, color='red', label=f"Regression
Line: y = {intercept:.2f} - {abs(slope):.2f}x")

# Add titles and labels
plt.title("Scatter Plot with Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
