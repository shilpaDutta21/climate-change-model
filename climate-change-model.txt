import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df=pd.read_csv("C://Users//Shilpa Kumari//Downloads//temps.csv")
df
# Display the first few rows of data
print(df.head(10))
# Check for any missing values in the dataset

if df.isnull().values.any():
    print("Data contains missing values. Consider handling them before proceeding.")
else:
    print("No missing values found.")
# Define the features (years) and target (temperature anomaly)
X = df[['year']]
y = df['temp_1']
# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Climate Change Model using Linear Regression')
plt.legend()
plt.show()
# Predict future temperature anomaly (e.g., for the year 2050)

future_year = np.array([[2050]])
future_temp = model.predict(future_year)
print(f'Predicted Temperature Anomaly in 2050: {future_temp[0]:.2f}°C')
