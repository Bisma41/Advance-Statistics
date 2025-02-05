import pandas as pd
import statsmodels.api as sm

# Dataset for animal weight prediction
data = {
    'Initial_Weight': [42, 33, 33, 45, 39, 36, 32, 41, 40, 38],
    'Feed_Weight': [272, 226, 259, 292, 311, 183, 173, 236, 230, 235],
    'Final_Weight': [95, 77, 80, 100, 97, 70, 50, 80, 92, 84]
}
df = pd.DataFrame(data)

# Independent variables
X = df[['Initial_Weight', 'Feed_Weight']]
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Dependent variable
Y = df['Final_Weight']

# Fit the multiple regression model
model = sm.OLS(Y, X).fit()

# Output the regression results
print(model.summary())

# Predict the final weight for an animal with initial weight 35kg and feed weight 250kg
new_data = pd.DataFrame({'const': [1], 'Initial_Weight': [35], 'Feed_Weight': [250]})
predicted_weight = model.predict(new_data)
print(f"Predicted Final Weight: {predicted_weight[0]:.2f} kg")

# Dataset for stopping distance experiment
data_sd = {
    'Speed': [35, 50, 65, 80, 95, 110],
    'Stopping_Distance': [16, 26, 41, 62, 88, 119]
}
df_sd = pd.DataFrame(data_sd)

# Create quadratic term
X_sd = pd.DataFrame({'Speed': df_sd['Speed'], 'Speed_Squared': df_sd['Speed']**2})
X_sd = sm.add_constant(X_sd)

# Dependent variable
Y_sd = df_sd['Stopping_Distance']

# Fit the multiple regression model
model_sd = sm.OLS(Y_sd, X_sd).fit()

# Output the regression results
print(model_sd.summary())

# Predict the stopping distance for a car traveling at 70 km/hr
new_data_sd = pd.DataFrame({'const': [1], 'Speed': [70], 'Speed_Squared': [70**2]})
predicted_distance = model_sd.predict(new_data_sd)
print(f"Predicted Stopping Distance: {predicted_distance[0]:.2f} meters")
