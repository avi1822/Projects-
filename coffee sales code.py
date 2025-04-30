import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('C:/Users/Sanika/Downloads/coffee_sales.csv.csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill missing numerical values with the median
if 'money' in data.columns:
    data['money'].fillna(data['money'].median(), inplace=True)
else:
    print("Column 'money' not found in the dataset.")

# Fill missing categorical values with the mode
if 'coffee_name' in data.columns:
    data['coffee_name'].fillna(data['coffee_name'].mode()[0], inplace=True)
else:
    print("Column 'coffee_name' not found in the dataset.")

# Convert 'date' to datetime type with error handling
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
else:
    print("'date' column is missing.")

# Removing Outliers
# Ensure the columns exist before attempting outlier removal
if 'money' in data.columns:
    z_scores = zscore(data[['money']])
    # Filter rows where Z-score is less than 3 for 'money'
    data = data[(np.abs(z_scores) < 3).all(axis=1)]
else:
    print("Column 'money' required for outlier removal is missing.")

# Feature Engineering
# Extract month and year from the 'date' column
if 'date' in data.columns:
    data['Month'] = data['date'].dt.month
    data['Year'] = data['date'].dt.year
else:
    print("'date' column is missing, unable to perform feature engineering.")

# Exploratory Data Analysis (EDA)
# Sales (money) over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Month', y='money', hue='Year')
plt.title('Monthly Sales Over Years')
plt.xlabel('Month')
plt.ylabel('Money')
plt.legend(title='Year')
plt.show()

# Sales by coffee type
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='coffee_name', y='money')
plt.title('Sales by Coffee Type')
plt.xlabel('Coffee Type')
plt.ylabel('Money')
plt.show()

# Machine Learning Modeling
# Splitting the Data
# Define features and target variable
X = data.drop(columns=['money', 'datetime', 'date'])  # Exclude irrelevant columns
y = data['money']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Simple Model
# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Interpret the model by looking at the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Summary
print("Steps completed successfully:")
print("1. Loaded and cleaned the coffee sales data.")
print("2. Conducted EDA to visualize sales trends.")
print("3. Prepared data for machine learning with categorical handling and train-test splitting.")
print("4. Trained a linear regression model to predict sales.")
print("5. Evaluated the model's performance with MSE and R^2 metrics.")
