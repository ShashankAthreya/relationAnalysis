#%%
# Import necessary libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from joblib import dump
#%%
# Load and preprocess the data
slpa_data = xr.open_dataset("/Users/sam/Downloads/MLP_Sonja/slpa_1950_2021.nc")
ssta_data = xr.open_dataset("/Users/sam/Downloads/MLP_Sonja/ssta_1950_2021.nc")

# %%
# Preprocessing
# Data Alignment: Ensure that both SLPA and SSTA data cover the same time period
start_date = max(slpa_data.time.min(), ssta_data.time.min())
end_date = min(slpa_data.time.max(), ssta_data.time.max())
slpa_data = slpa_data.sel(time=slice(start_date, end_date))
ssta_data = ssta_data.sel(time=slice(start_date, end_date))

# Removing Duplicates Times
slpa_data = slpa_data.sel(time=~slpa_data.indexes['time'].duplicated())
ssta_data = ssta_data.sel(time=~ssta_data.indexes['time'].duplicated())

# Missing Data Handling: Interpolate missing data using linear interpolation
slpa_data = slpa_data.interpolate_na(dim="time", method="linear")
ssta_data = ssta_data.interpolate_na(dim="time", method="linear")

# Temporal Averaging: Calculate 3-month moving averages to reduce noise
slpa_data = slpa_data.rolling(time=3, center=True).mean()
ssta_data = ssta_data.rolling(time=3, center=True).mean()

# Standardization and Flattening: Convert the data to numpy arrays and flatten the data
slpa_np = slpa_data["SLPA"].values
ssta_np = ssta_data["SSTA"].values
slpa_flat = slpa_np.reshape(slpa_np.shape[0], -1)
ssta_flat = ssta_np.reshape(ssta_np.shape[0], -1)

# Remove any remaining NaNs after temporal averaging (first and last values)
valid_indices = np.logical_not(np.isnan(slpa_flat).any(axis=1) | np.isnan(ssta_flat).any(axis=1))
slpa_flat = slpa_flat[valid_indices]
ssta_flat = ssta_flat[valid_indices]

# %%
# Train-test Split: Divide the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(slpa_flat, ssta_flat, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the MLP model
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Predict and evaluate the MLP model
y_pred = mlp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MLP Mean Squared Error: ", mse)
print("MLP R-squared: ", r2)

# Create a baseline model (Linear Regression)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression Mean Squared Error: ", mse_lr)
print("Linear Regression R-squared: ", r2_lr)

# Optimize hyperparameters

parameters = {
    "hidden_layer_sizes": [(50,), (100,), (150,)],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init": [0.001, 0.01, 0.1],
    "activation": ["relu", "tanh"],
}
mlp_gs = MLPRegressor(max_iter=1000, random_state=42)
grid_search = GridSearchCV(mlp_gs, parameters, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)

# Train the optimized MLP model
mlp_opt = grid_search.best_estimator_
dump(mlp_opt, 'best_model_mlp.joblib')
mlp_opt.fit(X_train, y_train)

# Predict and evaluate the optimized MLP model
y_pred_opt = mlp_opt.predict(X_test)
mse_opt = mean_squared_error(y_test, y_pred_opt)
r2_opt = r2_score(y_test, y_pred_opt)
print("Optimized MLP Mean Squared Error: ", mse_opt)
print("Optimized MLP R-squared: ", r2_opt)

# %%
# Visualize the results
plt.figure(figsize=(12, 6))

# MLP model
plt.subplot(2, 2, 1)
plt.title("MLP Predictions")
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
plt.xlabel("True SSTA")
plt.ylabel("Predicted SSTA")
plt.plot([-4, 4], [-4, 4], color="red", label="Ideal fit")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("MLP Error Distribution")
plt.hist(y_test - y_pred, bins=30, alpha=0.5)
plt.xlabel("Error")
plt.ylabel("Frequency")

# Linear Regression model
plt.subplot(2, 2, 3)
plt.title("Linear Regression Predictions")
plt.scatter(y_test, y_pred_lr, alpha=0.5, label="Predictions")
plt.xlabel("True SSTA")
plt.ylabel("Predicted SSTA")
plt.plot([-4, 4], [-4, 4], color="red", label="Ideal fit")
plt.legend()

plt.subplot(2, 2, 4)
plt.title("Linear Regression Error Distribution")
plt.hist(y_test - y_pred_lr, bins=30, alpha=0.5)
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Interpretation of Results

# Evaluation Metrics
print("MLP Mean Squared Error: ", mse)
print("MLP R-squared: ", r2)
print("Linear Regression Mean Squared Error: ", mse_lr)
print("Linear Regression R-squared: ", r2_lr)

# Visualizations
plt.figure(figsize=(12, 6))

# MLP model
plt.subplot(2, 2, 1)
plt.title("MLP Predictions")
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
plt.xlabel("True SSTA")
plt.ylabel("Predicted SSTA")
plt.plot([-4, 4], [-4, 4], color="red", label="Ideal fit")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("MLP Error Distribution")
plt.hist(y_test - y_pred, bins=30, alpha=0.5)
plt.xlabel("Error")
plt.ylabel("Frequency")

# Linear Regression model
plt.subplot(2, 2, 3)
plt.title("Linear Regression Predictions")
plt.scatter(y_test, y_pred_lr, alpha=0.5, label="Predictions")
plt.xlabel("True SSTA")
plt.ylabel("Predicted SSTA")
plt.plot([-4, 4], [-4, 4], color="red", label="Ideal fit")
plt.legend()

plt.subplot(2, 2, 4)
plt.title("Linear Regression Error Distribution")
plt.hist(y_test - y_pred_lr, bins=30, alpha=0.5)
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Optimized Model Parameters
print("Best parameters: ", grid_search.best_params_)

# %%
