import pandas as pd
import numpy as np

# Step 1: Data Collection
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Marks': [85, 92, 78, 88, 95],
    'CGPA': [3.5, 4.0, 2.8, 3.8, 4.0],
    'Percentage': [85, 92, 78, 88, 95]
}

df = pd.DataFrame(data)

print("Step 1: Data Collection")
print(df)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 3: Model Selection and Training
X = df[['Marks', 'CGPA']]
y = df['Percentage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nStep 3: Model Selection and Training")
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

       
    