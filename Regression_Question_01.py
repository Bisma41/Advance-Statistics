import numpy as np
from scipy.stats import linregress

# Given data points

x = np.array([3, 1, 3, 5]) # Independent variable
y = np.array([5, 8, 6, 4]) # Dependent variable

# Calculate the slope and intercept using linregress

slope, intercept, r_value, p_value, std_err =
linregress(x, y)

# Formulate the regression equation

regression_equation = f"y = {intercept:.2f} +
{slope:.2f}x"

# Output results

print("Slope:", slope)
print("Intercept:", intercept)
print("Regression Equation:", regression_equation)
