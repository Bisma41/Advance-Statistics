import pandas as pd
import statsmodels.api as sm

# Dataset
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
