import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)
x = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

# Visualize dataset
plt.scatter(x, y, color='yellow', label='Data')
plt.title("SYNTHETIC DATASET")
plt.xlabel("FEATURE")
plt.ylabel("TARGET")
plt.legend()
plt.show()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the model
regressor = DecisionTreeRegressor(max_depth=4, random_state=42)
regressor.fit(x_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot model prediction
x_grid = np.arange(min(x), max(x), 0.01)[:, np.newaxis]
y_grid = regressor.predict(x_grid)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='yellow', label='Data')
plt.plot(x_grid, y_grid, color='red', label='Model Prediction')
plt.title("DECISION TREE REGRESSION")
plt.xlabel('FEATURE')
plt.ylabel('TARGET')
plt.legend()
plt.show()

# Visualize the decision tree
print("Now showing the Decision Tree Structure...")
plt.figure(figsize=(20, 10))
plot_tree(
    regressor,
    feature_names=["Feature"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Structure")
plt.show()
