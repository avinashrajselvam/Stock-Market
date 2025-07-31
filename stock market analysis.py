import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
print("Current working directory:", os.getcwd())
csv_path = 'E:/stock market analysis/Stocks.csv'  # Change this if needed

# Check if file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

data = pd.read_csv(csv_path)

# Check required columns
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
for col in required_cols:
    if col not in data.columns:
        raise KeyError(f"Missing column: {col}")

data['Date'] = pd.to_datetime(data['Date'])

# Step 2: Check for missing values
print("Missing values:\n", data.isnull().sum())

# Step 3: Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()

# Step 4: Distribution of closing prices
plt.figure(figsize=(8, 5))
plt.hist(data['Close'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.title('Closing Price Distribution')
plt.show()

# Step 5: Total Volume by Ticker
ticker_volume = data.groupby('Ticker')['Volume'].sum()
ticker_volume.plot(kind='bar', figsize=(8, 5), color='orange')
plt.xlabel('Ticker')
plt.ylabel('Total Volume')
plt.title('Total Volume by Ticker')
plt.show()

# Step 6: Volume vs Closing Price
plt.figure(figsize=(8, 5))
plt.scatter(data['Volume'], data['Close'], alpha=0.5)
plt.xlabel('Volume')
plt.ylabel('Closing Price')
plt.title('Volume vs. Closing Price')
plt.show()

# Step 7: Boxplot of Closing Price
plt.figure(figsize=(8, 5))
plt.boxplot(data['Close'])
plt.ylabel('Closing Price')
plt.title('Closing Price Boxplot')
plt.show()

# Step 8: Feature Engineering
data['MA7'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7).mean())
data['MA14'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=14).mean())
data['MA30'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=30).mean())
data['Daily_Return'] = data.groupby('Ticker')['Close'].pct_change()
data['Volatility_7'] = data.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(window=7).std())
data['Volatility_14'] = data.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(window=14).std())

# Step 9: Remove NaNs
data_cleaned = data.dropna().copy()

# Step 10: Prepare features and target
features = ['Open', 'High', 'Low', 'Volume', 'MA7', 'MA14', 'MA30', 'Volatility_7', 'Volatility_14']
X = data_cleaned[features]
y = data_cleaned['Close']

# Step 11: Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 12: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 13: Predict and evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model RMSE: {rmse}")
print(f"Model R2 Score: {r2}")