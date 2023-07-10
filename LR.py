import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract input and output variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create linear regression model
model = LinearRegression()

# Train the model using the input and output variables
model.fit(X, y)

# Print the coefficients and intercept of the linear regression model
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Plot the data points
plt.scatter(X, y, color='blue')

# Plot the regression line
plt.plot(X, model.predict(X), color='red')

# Add labels and title to the plot
plt.xlabel('Input Variable')
plt.ylabel('Output Variable')
plt.title('Linear Regression')
plt.show()
