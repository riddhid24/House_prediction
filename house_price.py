# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

# Load the Boston Housing Prices dataset
boston = load_boston()
data = pd.DataFrame(data=np.c_[boston.data, boston.target], columns=np.append(boston.feature_names, "PRICE"))

# Data preprocessing
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler(),
    VarianceThreshold(),
    TransformedTargetRegressor(
        regressor=Ridge(alpha=0.1),
        transformer=StandardScaler()
    )
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Plot the predicted vs. actual house prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

# Learning Curves
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
)

train_mse_mean = -np.mean(train_scores, axis=1)
test_mse_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mse_mean, label='Training MSE')
plt.plot(train_sizes, test_mse_mean, label='Validation MSE')
plt.xlabel("Training Set Size")
plt.ylabel("Negative MSE (Smaller is Better)")
plt.title("Learning Curves")
plt.legend()
plt.show()
