import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate clinical lab data
data = {
    'Glucose': [90, 110, 130, 85, 100, 120, 140, 95, 105, 125],
    'Cholesterol': [180, 210, 190, 170, 200, 220, 230, 175, 195, 205],
    'Hemoglobin': [13.5, 14.2, 12.9, 13.0, 13.8, 14.5, 12.7, 13.2, 13.9, 14.0],
    'Risk_Score': [2.5, 3.8, 4.2, 2.1, 3.0, 4.5, 4.8, 2.3, 3.2, 4.0]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Glucose', 'Cholesterol', 'Hemoglobin']]
y = df['Risk_Score']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot predicted vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='darkblue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Risk Score")
plt.ylabel("Predicted Risk Score")
plt.title("Actual vs Predicted Risk Scores")
plt.grid()
plt.show()

# Visualize the decision tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree for Clinical Lab Prediction")
plt.show()
