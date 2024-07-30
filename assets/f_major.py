import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate a random regression problem
X, y = make_regression(n_samples=1000, n_features=100, noise=0.1, random_state=16)
X, y = X.astype(np.float32), y.astype(np.float32)

# print("y:\n", y)
print("Before np.asfortranarray(X)")
print("X:\n", X)
print("X shape:\n", X.shape)
print("X strides:\n", X.strides)

# Convert X to column-major order
X_col_major = np.asfortranarray(X)

# Verify the order
print("After np.asfortranarray(X)")
print("X_col_major:\n", X_col_major)
print("X_col_major shape:\n", X_col_major.shape)
print("X_col_major strides:\n", X_col_major.strides)

#####
# sklearn is optimized for row-major computation, so no benefit from f major
#####
# Fit the model
# model = RandomForestRegressor()
# model.fit(X_col_major, y)
# Make predictions
# predictions = model.predict(X_col_major)
# print("Predictions:\n", predictions[:5])
