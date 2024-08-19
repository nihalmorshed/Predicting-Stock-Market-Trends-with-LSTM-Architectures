
# Predictive Modeling of Stock Market Trends with Stacked LSTM Architectures

The project demonstrates the use of a Recurrent Neural Networks(particularly the LSTMs) to predict stock prices using historical data. The model is built using Long Short-Term Memory (LSTM) layers, which are well-suited for time series prediction tasks.

## Dataset

The dataset used in this project is the **Google Stock Price** dataset, which contains historical stock prices. The data is imported from a CSV file and is preprocessed before being fed into the model.

## Data Preprocessing

- **Feature Scaling**: The dataset's features are scaled using the MinMaxScaler to normalize the data within the range of 0 to 1.
- **Data Structure Creation**: A data structure with 60 timesteps and 1 output is created to provide sequential data to the LSTM layers. This means that the model will consider the past 60 days' stock prices to predict the next day's price.
- **Reshaping**: The input data is reshaped into a 3D structure, which is required by the LSTM layers in Keras.

## Model Architecture

The RNN model consists of the following layers:
- **LSTM Layers**: Three LSTM layers are used with 50 units each. These layers are responsible for capturing the temporal dependencies in the data.
- **Dropout Regularization**: A dropout rate of 20% is applied after each LSTM layer to prevent overfitting by randomly setting a fraction of input units to zero at each update during training.
- **Dense Layer**: A fully connected Dense layer is added as the output layer with a single neuron to predict the stock price.

## Training the Model

The model is compiled using the **Adam optimizer** and the **Mean Squared Error (MSE)** loss function. The training process involves fitting the model to the preprocessed training data. 

- **Training Loss**: Around 0.0013 

## Results & Discussion

The model's performance can be evaluated using a separate test set. Common metrics to consider include MSE, RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error).

<img src="/Stock Prediction.png" alt="Stock Price Prediction" width="500" align="center">



### Potential Improvements
- **Tuning LSTM Units**: Adjusting the number of LSTM units and layers may lead to better performance.
- **Increasing Dropout Rate**: If overfitting is detected, increasing the dropout rate might help improve the model's generalization.
- **Experimenting with Optimizers**: Trying different optimizers like RMSprop or SGD could also affect performance.
- **Hyperparameter Tuning**: Conduct a grid search to find the optimal hyperparameters for the LSTM layers.

## How to Use

1. Clone the repository.
2. Install the necessary dependencies from the `requirements.txt`.
3. Run the Jupyter notebook to train the model and predict stock prices.

## Conclusion

This RNN model, enhanced with LSTM layers and dropout regularization, provides a robust approach for time series forecasting, particularly for stock price prediction. We can see our predictions lag behind the actual values because our model cannot react to fast non-linear changes. On the other hand, for the parts of the predictions containing smooth changes, our model will react pretty well and manage to follow the upward and downward trends. With further tuning and evaluation, the model's accuracy can be improved to make more precise predictions.
