import pandas as pd
import statsmodels.api as sm

# Dataset from Example 11.8
data = {
    'Temperature': [200, 200, 250, 250, 300, 300],
    'Concentration': [0.2, 0.4, 0.2, 0.4, 0.2, 0.4],
    'Yield': [40, 45, 50, 55, 60, 65]
}
df = pd.DataFrame(data)

# Independent variables
X = df[['Temperature', 'Concentration']]
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Dependent variable
Y = df['Yield']

# Fit the multiple regression model
model = sm.OLS(Y, X).fit()

# Output the regression results
print(model.summary())

# Predict the yield for a temperature of 225°C and concentration of 0.3
new_data = pd.DataFrame({'const': [1], 'Temperature': [225], 'Concentration': [0.3]})
predicted_yield = model.predict(new_data)
print(f"Predicted Yield: {predicted_yield[0]:.2f}")
