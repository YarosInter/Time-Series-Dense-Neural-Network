import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from datetime import datetime
from UsefulFunctions import data
import cufflinks as cf
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.graph_objs as go



def compute_strategy_returns(y_test, y_pred):
    """
    Computes the strategy returns by comparing real percent changes with model predictions.

    This function creates a DataFrame that includes the actual percent changes (`y_test`) and the model's
    predicted values (`y_pred`). It then calculates the directional positions based on the actual and predicted
    values, and computes the returns by multiplying the real percent change by the predicted position.

    Args:
        y_test (Series or DataFrame): The actual percent changes (real values).
        y_pred (Series or DataFrame): The predicted values from the model.

    Returns:
        DataFrame: A DataFrame containing the actual percent changes, predicted values, 
                   real positions, predicted positions, and computed returns.
    """

    # Initialize a DataFrame with the actual percent changes and add the model's predictions
    df = pd.DataFrame(y_test)
    df["prediction"] = y_pred

    # Add columns for the real and predicted directional positions
    df["real_position"] = np.sign(df["pct_change"])
    df["pred_position"] = np.sign(df["prediction"])

    # Calculate the strategy returns by multiplying the actual percent change by the predicted position
    # Note: Predictions are based on the previous bar's data, so no additional shift is needed
    df["returns"] = df["pct_change"] * df["pred_position"]

    return df



def plot_test_returns_cufflinks(returns, name=" "):
    """
    Plots the cumulative returns of one or multiple test sets along with a break-even line.

    This function takes a 2D array-like object `returns` and converts it into a pandas DataFrame, 
    or takes an already pandas DataFrame with multiple columns of returns, one for each test,
    plots the cumulative returns for each column as a separate line plot. It also includes 
    a horizontal break-even line at y = 0 to indicate no profit or loss.

    Parameters:
    -----------
    returns : array-like
        A 2D array-like object (such as a list of lists or a pandas DataFrame) where each column 
        represents the returns of a different test set over time.

    name : string, optional
        A string to be used in the plot title to describe the dataset or the specific test set. 
        Default is an empty string, which will result in the title "Cumulative Returns" without 
        additional description.

    Returns:
    --------
    None
        The function displays the plot using Plotly but does not return any value.
    """

    # Initialize Plotly offline mode
    pyo.init_notebook_mode(connected=True)

    df = pd.DataFrame(returns)

    # Create a subplot
    fig = make_subplots(rows=1, cols=1)

    # Add cumulative returns data for test set
    for i in range(len(df.columns)):
        trace_test = go.Scatter(x=df.index, y=df.iloc[:, i].cumsum() * 100, mode="lines", name=f"Retruns Test {i}")
        fig.add_trace(trace_test)
    
    # Add break-even line
    trace_break_even = go.Scatter(x=df.index, y=[0] * len(df.index), mode="lines", name="Break-even", line=dict(color="red"))
    fig.add_trace(trace_break_even)

    # Set layout
    fig.update_layout(title=f"Cumulative Returns {name}", xaxis_title="Time", yaxis_title="P&L in %", height=450)

    # Display the plot
    pyo.iplot(fig)
    print(f"Profits : {'%.2f' % (df.cumsum().iloc[-1].sum() * 100)}%")



def plot_test_returns(returns, legend=True, name=" "):
    """
    Plots the cumulative percentage returns from a trading strategy.

    This function takes a series or dataframe of trading strategy returns, computes the cumulative sum,
    and plots it as a percentage. The plot visualizes the profit and loss (P&L) over time.

    Args:
        returns_serie (pandas.Series): A series containing the returns from the trading strategy.

    Returns:
        None: The function generates and displays a plot.
    """
    
    # Plot cumulative returns as a percentage
    (np.cumsum(returns) * 100).plot(figsize=(15, 5), alpha=0.65)
    
    # Draw a red horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='-', linewidth=1)
    
    # Set labels and title
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('P&L in %', fontsize=20)
    plt.title(f'Cumulative Returns {name}', fontsize=20)
    plt.legend().set_visible(legend)
    print(f"Profits : {'%.2f' % (returns.cumsum().iloc[-1].sum() * 100)}%")

    # Display the plot
    plt.show()

    

def vectorize_backtest_returns(returns, anualization_factor, benchmark_asset=".US500Cash", mt5_timeframe=mt5.TIMEFRAME_H1):
    """
    Computes and prints the Sortino Ratio, Beta Ratio, and Alpha Ratio for a given set of returns series.
    
    Parameters:
    - returns: Series of returns from a strategy.
    - anualization_factor: Factor used to annualize returns.
    - benchmark_asset: The benchmark asset for comparison (default is S&P 500).
    - mt5_timeframe: Timeframe for pulling benchmark data (default is 1H).

    Note: The timeframe for benchmark data must match the timeframe of the strategy's returns for accurate results.
    
    Returns:
    None
    """

    ### Computing Sortino Ratio ###

    # Sortino Ratio is being calculated without a Risk-Free Rate
    mean_return = np.mean(returns)
    downside_deviation = np.std(returns[returns < 0])
    
    # Number of 15-minute periods in a year
    periods_per_year = anualization_factor * 252
    
    # Annualizing the mean return and downside deviation
    annualized_mean_return = mean_return * periods_per_year
    annualized_downside_deviation = downside_deviation * np.sqrt(periods_per_year)
    
    # Calculating the annualized Sortino ratio
    annualized_sortino = annualized_mean_return / annualized_downside_deviation
    
    print(f"Sortino Ratio: {'%.3f' % annualized_sortino}")
    if annualized_sortino > 0:
        print("- Positive Sortino (> 0): The investment’s returns exceed the target return after accounting for downside risk.\n")
    else:
        print("- Negative Sortino (< 0): The investment’s returns are less than the target return when considering downside risk.\n")

    ### Computing Beta Ratio ###

    print("***Asset for Benchamark is S&P500***\n")

    # Fetching the oldest date from the X_test set to pull data from the same date for the S&P 500
    date = returns.index.min()
    
    # Extracting the year, month, and day from the date
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    min = date.minute
    sec = date.second
    
    # Pulling S&P 500 data from the specified date and time
    sp500_data = data.get_rates(".US500Cash", mt5.TIMEFRAME_H1, from_date=datetime(year, month, day))
    sp500_data = sp500_data[["close"]]
    
    # Computing the returns on the S&P 500
    sp500_data["returns"] = sp500_data["close"].pct_change(1)
    sp500_data.drop("close", axis=1, inplace=True)
    sp500_data.dropna(inplace=True)
    
    # Concatenate values between the returns in the predictions and the returns in the S&P 500
    val = pd.concat((returns, sp500_data["returns"]), axis=1)
    
    # Changing column names to identify each one
    val.columns.values[0] = "Returns Pred"
    val.columns.values[1] = "Returns SP500"
    val.dropna(inplace=True)

    # Calculating Beta Ratio
    covariance_matrix = np.cov(val.values, rowvar=False)
    covariance = covariance_matrix[0][1]
    variance = covariance_matrix[1][1]
    beta = covariance / variance
    
    print(f"Beta Ratio: {'%.3f' % beta}")
    if beta == 1:
        print("- Beta ≈ 1: The asset moves in line with the market.\n")
    elif beta < 1:
        print("- Beta < 1: The asset is less volatile than the market (considered less risky).\n")
    else:
        print("- Beta > 1: The asset is more volatile than the market (higher potential return but also higher risk).\n")

    ### Computing Alpha Ratio ###

    alpha = (anualization_factor * 252 * mean_return * (1 - beta)) * 100
    
    print(f"Alpha Ratio: {'%.3f' % alpha}")
    if alpha > 0:
        print("- Positive Alpha (> 0): The investment outperformed the market.")
    else:
        print("- Negative Alpha (< 0): The investment underperformed the market.")



def compute_model_accuracy_cufflinks(real_positions, predicted_positions):
    """
    Computes and displays the accuracy of predicted positions compared to real positions.

    Parameters:
    real_positions (list or array-like): The actual positions.
    predicted_positions (list or array-like): The positions predicted by the model.

    Returns:
    pd.DataFrame: A DataFrame containing the real positions, predicted positions, and accuracy (1 for correct, 0 for incorrect).
    
    Displays:
    - Counts of correct and incorrect predictions.
    - Histogram showing the distribution of accuracy values.
    - Model accuracy percentage.
    """
    
    import pandas as pd
    import numpy as np
    import cufflinks as cf

    # Initialize Cufflinks in offline mode
    cf.go_offline()

    # Creating Dataframe with real positions and predicted positions
    df_accuracy = pd.DataFrame(real_positions, columns=["real_position"])
    df_accuracy["pred_position"] = predicted_positions
    
    # Assigning 1 if the position forecasted is equal to the real position and 0 otherwise
    df_accuracy["accuracy"] = np.where(df_accuracy["real_position"] == df_accuracy["pred_position"], 1, 0)

    # Count the occurrences of each unique accuracy value in the 'accuracy' column and store the result in 'accuracy'
    accuracy = df_accuracy["accuracy"].value_counts()

    # Printing explanation for the counts of 0 and 1 in the 'accuracy' column
    print("Counts of 0 indicate instances where the predicted position did not match the real position.")
    print("Counts of 1 indicate instances where the predicted position matched the real position.\n")
    print(accuracy)

    # Total counts of occurrences where the model was right (number assigned 1) divided by the total number of predictions
    model_accuracy = accuracy[1] / len(df_accuracy)
    print(f"\nModel has an accuracy of: {model_accuracy * 100:.2f}%")
    
    # Plotting the accuracy of the model in a histogram using the dynamic plot with Cufflinks
    df_accuracy["accuracy"].iplot(
        kind="hist",       
        xTitle="Prediction Resul", 
        yTitle="Counts",
        title="Model Accuracy",
        bargap=0.2,
        theme="white",         
        colors=["blue"],
        #layout=dict(height=400)
    )
        
    return df_accuracy



def compute_model_accuracy(real_positions, predicted_positions):
    """
    Computes and displays the accuracy of predicted positions compared to real positions.

    Parameters:
    real_positions (list or array-like): The actual positions.
    predicted_positions (list or array-like): The positions predicted by the model.

    Returns:
    pd.DataFrame: A DataFrame containing the real positions, predicted positions, and accuracy (1 for correct, 0 for incorrect).
    
    Displays:
    - Counts of correct and incorrect predictions.
    - Bar plot showing the distribution of accuracy values with a gap between bars.
    - Model accuracy percentage.
    """
    
    # Creating DataFrame with real positions and predicted positions
    df_accuracy = pd.DataFrame({'real_position': real_positions})
    df_accuracy["pred_position"] = predicted_positions
    
    # Assigning 1 if the position forecasted is equal to the real position and 0 otherwise
    df_accuracy["accuracy"] = np.where(df_accuracy["real_position"] == df_accuracy["pred_position"], 1, 0)

    # Count the occurrences of each unique accuracy value in the 'accuracy' column
    accuracy = df_accuracy["accuracy"].value_counts()

    # Printing explanation for the counts of 0 and 1 in the 'accuracy' column
    print("Counts of 0 indicate instances where the predicted position did not match the real position.")
    print("Counts of 1 indicate instances where the predicted position matched the real position.\n")
    print(accuracy)

    # Total counts of occurrences where model was right (number assigned 1) divided into the total number of predictions
    model_accuracy = accuracy[1] / len(df_accuracy)
    print(f"\nModel has an accuracy of: {model_accuracy * 100:.2f}%")

    # Create a bar plot with a gap between bars
    plt.bar(accuracy.index, accuracy.values, width=0.8)  # width adjusted for bar gap
    plt.xticks([0, 1], labels=['Incorrect = 0', 'Correct = 1'])
    plt.title("Model Accuracy", fontsize=20)
    plt.ylabel("Counts", fontsize=15)
    plt.xlabel("Prediction Resul", fontsize=15)
    plt.show()

    return df_accuracy



def plot_drawdown_cufflinks(return_series):
    """
    Computes and visualizes the drawdown of a strategy based on its return series.

    Parameters:
    return_series (pd.Series): A pandas Series containing the return series of the strategy. 
                               Each value represents the return for a specific period.

    Displays:
    - A plot showing the drawdown over time as a filled area chart.
    - The maximum drawdown percentage is printed to the console.

    Notes:
    - The function assumes the return series is cumulative and starts at zero.
    - NaN values in the return series are dropped before computation.
    - If the return series is empty or contains only NaN values, no plot will be generated.
    """
    
    if return_series.dropna().empty:
        print("The return series is empty or contains only NaN values.")
        return

    # Compute cumulative return
    cumulative_return = return_series.dropna().cumsum() + 1

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_return)

    # Computing the drawdown
    drawdown = cumulative_return / running_max - 1

    fig = make_subplots(rows=1, cols=1)

    trace_test = go.Scatter(x=drawdown.index, 
                            y=drawdown * 100, 
                            mode="lines", 
                            fill="tozeroy", 
                            name="Drawdown", 
                            line=dict(color="red"),
                            fillcolor="rgba(255, 0, 0, 0.8)") # Red color with 0.8 opacity
                            
    fig.add_trace(trace_test)
    
    fig.update_layout(title="Strategy Drawdown", xaxis_title="Time", yaxis_title="Drawdown in %", height=450, showlegend=True)
    pyo.iplot(fig)

    maximum_drawdown = np.min(drawdown) * 100
    print(f"Max Drawdown: {'%.2f' % maximum_drawdown}%")



def plot_drawdown(return_series, name=" "):
    """
    Computes and visualizes the drawdown of a strategy based on its return series.

    Parameters:
    return_series (pd.Series): A pandas Series containing the return series of the strategy. 
                               Each value represents the return for a specific period.

    Displays:
    - A plot showing the drawdown over time as a filled area chart.
    - The maximum drawdown percentage is printed to the console.

    Notes:
    - The function assumes the return series is cumulative and starts at zero.
    - NaN values in the return series are dropped before computation.
    - If the return series is empty or contains only NaN values, no plot will be generated.
    """
    
    if return_series.dropna().empty:
        print("The return series is empty or contains only NaN values.")
        return

    # Compute cumulative return
    cumulative_return = return_series.dropna().cumsum() + 1

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_return)

    # Computing the drawdown
    drawdown = cumulative_return / running_max - 1

    plt.figure(figsize=(15, 4))
    plt.fill_between(drawdown.index, drawdown * 100, 0, drawdown, color="red", alpha=0.70)
    plt.title(f"Strategy Drawdown {name}", fontsize=20)
    plt.ylabel("Drawdown %", fontsize=15)
    plt.xlabel("Time")
    plt.show()

    maximum_drawdown = np.min(drawdown) * 100
    print(f"Max Drawdown: {'%.2f' % maximum_drawdown}%")



def compute_drawdown(returns):
    # Compute cumulative return
    cumulative_return = returns.dropna().cumsum() + 1

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_return)

    # Computing the drawdown
    drawdown = cumulative_return / running_max - 1
    return drawdown



def create_model_checkpoint(model_name, save_path="model_experiments"):
    """
    Creates a ModelCheckpoint callback to save the best-performing version of a model during training.
    
    Args:
        model_name (str): The name of the model to be used for saving the file.
        save_path (str, optional): The directory path where the model file will be saved. Defaults to "model_experiments".
    
    Returns:
        ModelCheckpoint: A callback that saves the model with the lowest validation loss.
    
    Notes:
        - The checkpoint saves the model in the provided directory (`save_path`) with the format "{model_name}.keras".
        - Only the model with the best validation loss is saved.
    """
    # Ensure the directory exists
    #os.makedirs(save_path, exist_ok=True)
    
    return ModelCheckpoint(
        filepath=os.path.join(save_path, f"{model_name}.keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True
    )


def run_dnn(p_model_name, layers=3, p_epochs=50, lr=0.001, ptn=5):
    """
    Builds, compiles, and trains a deep neural network (DNN) model with specified parameters.

    Args:
        p_model_name (str): Name of the model to be used for saving and tracking.
        layers (int, optional): Number of hidden layers to include in the model. Defaults to 3.
        p_epochs (int, optional): Number of epochs for training the model. Defaults to 50.
        lr (float, optional): Learning rate for the Adam optimizer. Defaults to 0.001.
        ptn (int, optional): Patience parameter for EarlyStopping, which controls how many epochs to wait for improvement before stopping. Defaults to 5.

    Returns:
        History: Training history of the model, including loss and metric values.

    Notes:
        - The model includes Dropout layers for regularization and uses ReLU activations in hidden layers.
        - EarlyStopping is used to stop training if the validation loss does not improve after a specified number of epochs (`ptn`).
        - ModelCheckpoint saves the model with the best validation loss during training.
        - The model is compiled using Mean Absolute Error (MAE) as the loss function and Adam optimizer with a specified learning rate (`lr`).
        - Validation data is used during training to monitor performance on unseen data.
    """
    
    hidden_layers = layers 
    p_model_name = Sequential(name=p_model_name)

    # Input layer
    p_model_name.add(Dense(64, input_shape=(X_train_scaled.shape[1],), name="input_layer"))

    # hidden layers
    for i in range(0, hidden_layers):
        p_model_name.add(Dense(64, activation="relu", name=f"hidden_layer_{i}"))
        p_model_name.add(Dropout(0.25, name=f"dropout_layer_{i}"))

    # Output layer
    p_model_name.add(Dense(1, activation="linear", name="output_layer"))
    
    checkpoint_callback = create_model_checkpoint(p_model_name.name)
    early_stopping = EarlyStopping(monitor="val_loss", patience=ptn, verbose=1)

    # Compiling the model
    p_model_name.compile(loss="mae", optimizer=Adam(learning_rate=lr), metrics=["mae","mse"])

    # Training the model
    history = p_model_name.fit(X_train_scaled, np.sign(y_train), 
                                   validation_data=(X_val_scaled, np.sign(y_val)), 
                                   batch_size=128, 
                                   epochs=p_epochs,
                                   verbose=0,
                                   callbacks=[checkpoint_callback, early_stopping]
                                  )

    return history