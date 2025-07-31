# Stock Market Analysis Script Documentation

## Overview

This Python script performs exploratory data analysis (EDA) and builds a linear regression model to predict stock closing prices using historical stock market data. It includes data loading, validation, visualization, feature engineering, and model training/evaluation. The script is designed to work with a CSV dataset containing stock market data and generates various visualizations to analyze trends and relationships.

## File Information

- **Filename**: `stock_analysis.py`
- **Language**: Python 3
- **Dependencies**:
  - `os`: For file path operations.
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
  - `matplotlib.pyplot`: For plotting visualizations.
  - `seaborn`: For enhanced visualization (e.g., heatmap).
  - `scikit-learn`: For machine learning tasks (train-test split, scaling, linear regression, and metrics).

## Script Structure and Functionality

The script is divided into 13 steps, each performing a specific task in the data analysis and modeling pipeline.

### Step 1: Load the Dataset
- **Description**: Loads the stock market dataset from a CSV file.
- **Input**: A CSV file (`Stocks.csv`) located at the path specified in `csv_path`.
- **Validation**:
  - Checks if the file exists using `os.path.exists()`.
  - Verifies the presence of required columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Ticker`.
- **Processing**: Converts the `Date` column to a datetime format using `pd.to_datetime()`.
- **Output**: Prints the current working directory and loads the data into a pandas DataFrame (`data`).

### Step 2: Check for Missing Values
- **Description**: Identifies and reports missing values in the dataset.
- **Output**: Prints the count of missing values for each column using `data.isnull().sum()`.

### Step 3: Correlation Matrix Heatmap
- **Description**: Visualizes the correlation between numerical features.
- **Method**: Uses `seaborn.heatmap()` to plot a correlation matrix with annotations.
- **Output**: Displays a heatmap with a `coolwarm` colormap, showing correlation coefficients.

### Step 4: Distribution of Closing Prices
- **Description**: Analyzes the distribution of stock closing prices.
- **Method**: Plots a histogram of the `Close` column using `plt.hist()`.
- **Output**: Displays a histogram with 20 bins, colored skyblue with black edges.

### Step 5: Total Volume by Ticker
- **Description**: Summarizes the total trading volume for each stock ticker.
- **Method**: Groups the data by `Ticker` and sums the `Volume` column, then plots a bar chart.
- **Output**: Displays a bar plot with tickers on the x-axis and total volume on the y-axis.

### Step 6: Volume vs. Closing Price
- **Description**: Examines the relationship between trading volume and closing price.
- **Method**: Creates a scatter plot using `plt.scatter()` with `Volume` on the x-axis and `Close` on the y-axis.
- **Output**: Displays a scatter plot with semi-transparent points (`alpha=0.5`).

### Step 7: Boxplot of Closing Price
- **Description**: Visualizes the distribution and outliers of closing prices.
- **Method**: Plots a boxplot of the `Close` column using `plt.boxplot()`.
- **Output**: Displays a boxplot showing the median, quartiles, and potential outliers.

### Step 8: Feature Engineering
- **Description**: Creates additional features to improve model performance.
- **Features Added**:
  - `MA7`: 7-day moving average of closing prices.
  - `MA14`: 14-day moving average of closing prices.
  - `MA30`: 30-day moving average of closing prices.
  - `Daily_Return`: Daily percentage change in closing prices.
  - `Volatility_7`: 7-day rolling standard deviation of daily returns.
  - `Volatility_14`: 14-day rolling standard deviation of daily returns.
- **Method**: Uses `pandas` groupby and rolling window functions to compute features per ticker.

### Step 9: Remove NaNs
- **Description**: Removes rows with missing values introduced by feature engineering.
- **Method**: Uses `data.dropna()` to create a cleaned DataFrame (`data_cleaned`).

### Step 10: Prepare Features and Target
- **Description**: Defines the feature set and target variable for modeling.
- **Features**: `Open`, `High`, `Low`, `Volume`, `MA7`, `MA14`, `MA30`, `Volatility_7`, `Volatility_14`.
- **Target**: `Close` (closing price).
- **Output**: Creates feature matrix `X` and target vector `y`.

### Step 11: Train-Test Split and Scaling
- **Description**: Splits data into training and testing sets and scales features.
- **Method**:
  - Splits data using `train_test_split` with 80% training and 20% testing (`test_size=0.2`, `random_state=42`).
  - Scales features using `StandardScaler` to standardize `X_train` and `X_test`.
- **Output**: Scaled training and testing feature sets (`X_train_scaled`, `X_test_scaled`).

### Step 12: Train Linear Regression Model
- **Description**: Trains a linear regression model to predict closing prices.
- **Method**: Fits a `LinearRegression` model on `X_train_scaled` and `y_train`.

### Step 13: Predict and Evaluate
- **Description**: Makes predictions on the test set and evaluates model performance.
- **Metrics**:
  - Root Mean Squared Error (RMSE): Measures prediction error magnitude.
  - R² Score: Measures the proportion of variance explained by the model.
- **Output**: Prints RMSE and R² score.

## Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Prepare the Dataset**:
   - Ensure the `Stocks.csv` file is available and contains the required columns.
   - Update the `csv_path` variable in the script to the correct file path:
     ```python
     csv_path = 'path/to/your/Stocks.csv'
     ```

3. **Run the Script**:
   ```bash
   python stock_analysis.py
   ```

4. **Expected Output**:
   - Console output:
     - Current working directory.
     - Missing value counts.
     - Model RMSE and R² score.
   - Visualizations:
     - Correlation matrix heatmap.
     - Closing price histogram.
     - Total volume by ticker bar plot.
     - Volume vs. closing price scatter plot.
     - Closing price boxplot.

## Notes

- **File Path**: Ensure the `csv_path` is correct to avoid `FileNotFoundError`.
- **Missing Values**: The script removes rows with NaNs after feature engineering, which may reduce the dataset size.
- **Visualizations**: Requires a graphical environment to display plots (e.g., Matplotlib backend).
- **Model**: The linear regression model assumes linear relationships; consider other models for complex patterns.
- **Dataset**: The script assumes the dataset is clean and properly formatted. Verify column names and data types before running.

## Error Handling

- **FileNotFoundError**: Raised if the CSV file is not found at the specified path.
- **KeyError**: Raised if any required column is missing from the dataset.

## Example Dataset Format

```csv
Date,Open,High,Low,Close,Volume,Ticker
2023-01-01,100.5,102.0,99.0,101.2,1000000,AAPL
2023-01-02,101.3,103.5,100.8,102.7,1200000,AAPL
...
```

## Future Improvements

- Add support for multiple models (e.g., Random Forest, XGBoost).
- Include time-series cross-validation for better model evaluation.
- Add more advanced feature engineering (e.g., technical indicators like RSI, MACD).
- Save visualizations to files for non-interactive environments.

## License

This script is part of a project licensed under the MIT License. See the [LICENSE](LICENSE) file for details.