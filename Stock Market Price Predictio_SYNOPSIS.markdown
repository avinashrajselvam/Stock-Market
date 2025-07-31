# Stock Market Analysis Synopsis

## Project Overview

**Author**: K S AVINASHRAJ  
**Email**: avinashgithub0707@gmail.com  
**GitHub**: [github.com/avinashrajselvam](https://github.com/avinashrajselvam)  
**Date**: July 31, 2025  

This project analyzes historical stock market data to uncover trends and predict closing prices using a linear regression model. Implemented in Python, it combines exploratory data analysis (EDA), feature engineering, and machine learning to provide insights into stock market behavior.

## Objectives

- Conduct comprehensive EDA to explore stock price distributions, correlations, and trading volumes.
- Engineer features to capture market trends and volatility.
- Develop and evaluate a linear regression model for predicting stock closing prices.
- Visualize key findings to support data-driven decision-making.

## Methodology

1. **Data Loading and Validation**:
   - Loads a CSV dataset (`Stocks.csv`) with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Ticker`.
   - Validates file existence and required columns, converting `Date` to datetime format.

2. **Exploratory Data Analysis**:
   - Generates visualizations:
     - Correlation matrix heatmap for numerical features.
     - Histogram of closing prices.
     - Bar plot of total volume by ticker.
     - Scatter plot of volume vs. closing price.
     - Boxplot for closing price outliers.
   - Utilizes Matplotlib and Seaborn for plotting.

3. **Feature Engineering**:
   - Creates features per ticker:
     - Moving averages (7, 14, 30 days).
     - Daily returns (percentage change in closing price).
     - Volatility (7 and 14-day standard deviation of returns).
   - Removes rows with missing values post-engineering.

4. **Modeling and Evaluation**:
   - Selects features: `Open`, `High`, `Low`, `Volume`, `MA7`, `MA14`, `MA30`, `Volatility_7`, `Volatility_14`.
   - Splits data (80% train, 20% test) and scales features using `StandardScaler`.
   - Trains a linear regression model and evaluates using RMSE and R² score.

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Dataset**: User-provided `Stocks.csv` with historical stock data

## Key Outcomes

- **Visual Insights**: Clear visualizations of price distributions, trading volumes, and correlations.
- **Feature Engineering**: Enhanced model performance with trend and volatility indicators.
- **Model Performance**: Baseline linear regression model with quantifiable metrics (RMSE, R²).
- **Reusability**: Modular script adaptable for other stock datasets.

## Future Scope

- Incorporate advanced models (e.g., Random Forest, XGBoost).
- Implement time-series cross-validation for robust evaluation.
- Add technical indicators (e.g., RSI, MACD).
- Support real-time data or multiple datasets.

## Repository

The complete project, including code, documentation, and presentation, is available at:  
[github.com/avinashrajselvam/Stock---Market](https://github.com/avinashrajselvam/Stock---Market)