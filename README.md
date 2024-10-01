# Time-Series Dense Neural Network

This repository focuses on time series analysis using a Dense Neural Network applied to financial data. The objective is to develop models capable of predicting financial trends and price directions over time using deep learning techniques.

## Project Overview

Time series analysis is essential for forecasting financial data, particularly for trading and investment strategies. This project utilizes historical financial data from the EURUSD forex pair and applies a dense neural network to predict future price movements. The network leverages TensorFlow's powerful deep learning tools to enhance performance and optimize predictions.

### Key Features

- **Data Retrieval**: Custom functions are used to fetch historical financial data from MetaTrader 5.
- **Data Processing**: The data is standardized and engineered for better model performance.
- **Model Development**: The Dense Neural Network is built using TensorFlow and Keras, consisting of:
  - Input layer with 64 units
  - 3 hidden layers with ReLU activation and Dropout layers for regularization
  - Output layer with a single unit to predict price direction
  - **EarlyStopping**: Implemented with a patience of 5 to prevent overfitting
  - **ModelCheckpoint**: Saves the best-performing model during training based on validation loss.
- **Model Evaluation**: Performance is assessed using MAE (Mean Absolute Error) and MSE (Mean Squared Error).
- **Visualization**: Plots include cumulative returns, drawdowns, and model performance metrics.

## Repository Structure

- **`data.py`**: Contains functions for retrieving and manipulating financial data, including:
  - `get_rates`: Fetches historical price data for a given symbol and timeframe.
  - `add_shifted_columns`: Adds lagged features to the DataFrame for time series predictions.
  - `split_data`: Splits the data into training, validation, and test sets.

- **`backtest.py`**: Provides functions for backtesting financial strategies, including:
  - `compute_strategy_returns`: Calculates cumulative returns for a given strategy.
  - `compute_drawdown`: Computes the drawdown for a strategy.
  - `plot_returns`: Visualizes cumulative returns.
  - `plot_drawdown`: Plots the drawdown of a strategy.
  - `compute_model_accuracy`: Computes the accuracy of model predictions.
  - `vectorize_backtest_returns`: Computes financial metrics such as Sortino, Beta, and Alpha ratios.

- **`Dense Layer Neural Network.ipynb`**: Jupyter Notebook containing the code for building, training, and evaluating the Dense Neural Network. The notebook walks through the following steps:
  - Data processing
  - Model creation and compilation using TensorFlow and Keras
  - Training the model with EarlyStopping and ModelCheckpoint callbacks
  - Evaluating the model on the validation set
  - Backtesting model performance on financial data

- **`.gitignore`**: Specifies files and directories to be ignored by Git.

- **`LICENSE`**: The MIT License under which this project is distributed.

- **`README.md`**: This file.

## Installation

To run this project, you need Python installed along with the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tensorflow`
- `MetaTrader5`

## Getting Started

1. **Clone the Repository for Viewing**: You may clone this repository to your local machine for personal review and educational purposes only:
   ```bash 
   git clone https://github.com/YarosInter/Time-Series-Dense-Neural-Network.git
   ```
2. **Install Dependencies**: Ensure all required Python packages are installed.

3. **Run the Notebook**: Open `Dense Layer Neural Network.ipynb` in Jupyter Notebook to explore the model and results.

4. **Data Retrieval**: Use `data.py` to fetch and preprocess financial data.

5. **Backtesting**: Use `backtest.py` to evaluate your strategy's performance using cumulative returns and drawdown analysis.

   
   
## Contributing

This repository is for personal or educational purposes only. Contributions and modifications are not permitted unless explicitly allowed. Feel free to reach out if you'd like to collaborate or contribute, let’s learn together!


## Disclaimer

The code in this repository is for educational or personal review only and is not licensed for use, modification, or distribution. 
This code is part of my journey in learning and experimenting with new ideas. It’s shared for educational purposes and personal review. Please feel free to explore, but kindly reach out for permission if you’d like to use, modify, or distribute any part of it.
