import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the housing dataset (Housing.csv must be present)
data = pd.read_csv('Housing.csv')

# Ensure the dataset contains the target column 'price'
if 'price' not in data.columns:
    raise ValueError("The dataset must have a 'price' column as the target variable.")

# Separate features and target
X = data.drop('price', axis=1)
y = data['price']

# (Optional) Convert categorical features to numeric using one-hot encoding if needed
X = pd.get_dummies(X)

# Optionally split the dataset (for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Save the trained regression model as reg_model.pkl
with open('reg_model.pkl', 'wb') as f:
    pickle.dump(reg, f)

print("Regression model trained and saved as reg_model.pkl")
print("Feature columns after one-hot encoding:", X.columns.tolist())
print("Number of features:", X.shape[1])
