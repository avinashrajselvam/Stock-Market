

# Stock Market Analysis

This project performs exploratory data analysis (EDA) and builds a linear regression model to predict stock closing prices using historical stock market data. The script generates visualizations to analyze trends, correlations, and distributions, and includes feature engineering to enhance model performance.

## Features

- **Data Loading and Validation**: Loads stock market data from a CSV file and checks for required columns and missing values.
- **Exploratory Data Analysis**:
  - Correlation matrix heatmap to identify relationships between numerical features.
  - Histogram of closing prices to show their distribution.
  - Bar plot of total trading volume by ticker.
  - Scatter plot of volume vs. closing price.
  - Boxplot of closing prices to detect outliers.
- **Feature Engineering**:
  - Calculates moving averages (7, 14, and 30 days).
  - Computes daily returns and volatility (7 and 14 days).
- **Modeling**: Trains a linear regression model to predict closing prices, with performance evaluation using RMSE and R² score.

## Requirements

To run this project, you need the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

The script expects a CSV file (`Stocks.csv`) with the following columns:
- `Date`: Date of the stock data (e.g., YYYY-MM-DD).
- `Open`: Opening price of the stock.
- `High`: Highest price of the stock during the day.
- `Low`: Lowest price of the stock during the day.
- `Close`: Closing price of the stock.
- `Volume`: Trading volume.
- `Ticker`: Stock ticker symbol.

Update the `csv_path` variable in the script to point to your dataset's location.

Example file path in the script:
```python
csv_path = 'E:/stock market analysis/Stocks.csv'
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Prepare the dataset**:
   - Place your `Stocks.csv` file in the appropriate directory.
   - Update the `csv_path` in the script if necessary.

3. **Run the script**:
   ```bash
   python stock_analysis.py
   ```

4. **Output**:
   - The script will print:
     - The current working directory.
     - Missing value counts in the dataset.
     - Model performance metrics (RMSE and R² score).
   - Visualizations will be displayed:
     - Correlation matrix heatmap.
     - Closing price distribution histogram.
     - Total volume by ticker bar plot.
     - Volume vs. closing price scatter plot.
     - Closing price boxplot.

## File Structure

- `stock_analysis.py`: Main Python script containing the analysis and modeling code.
- `Stocks.csv`: Input dataset (not included; user must provide).
- `README.md`: This file, providing project documentation.

## Notes

- Ensure the dataset path (`csv_path`) is correct to avoid `FileNotFoundError`.
- The script assumes the dataset has no missing values after cleaning (NaNs are dropped after feature engineering).
- The linear regression model uses scaled features for better performance.
- Visualizations are displayed using Matplotlib and Seaborn; ensure a graphical environment is available if running locally.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
