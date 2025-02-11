import pandas as pd
import numpy as np

def MSE(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def RMSE(y, y_pred):
    return np.sqrt(MSE(y, y_pred))

def MAE(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def singleBatchTrain(w, b, x, y, learning_rate):
    weight_derivative = -np.mean((y - (w*x+b)) * x * 2)
    bias_derivative = -np.mean((y - (w*x+b)) * 2)
    print(weight_derivative, '\n', bias_derivative, '\n')
    w -= weight_derivative * learning_rate
    b -= bias_derivative * learning_rate
    return w, b

def build_batch(df, batch_size):
    batch = df.sample(n=batch_size).copy()
    batch.set_index(np.arange(batch_size), inplace=True)
    return batch

def train_model(df, features, label, epochs, batch_size, learning_rate):
    hist = {"root_mean_squared_error": []}
    w = 0
    b = 0
    for i in range(epochs):
        for j in range(int(len(df[features]) / batch_size)):
            batch = build_batch(df, batch_size)
            x = batch[features].values
            y = batch[label].values
            w, b = singleBatchTrain(w, b, x, y, learning_rate)
            y_pred = x * w + b
            rmse = RMSE(y, y_pred)
            hist["root_mean_squared_error"].append(rmse)
        print(f"Epoch {i + 1}/{epochs}, RMSE: {RMSE(y, y_pred)}, w: {w}, b: {b}")

    return w, b

train_data = pd.read_csv('train.csv')
feature = 'Current_Size'
label = 'Voltage'
epochs = 10
batch_size = 100
learning_rate = 0.01
w, b = train_model(train_data, feature, label, epochs, batch_size, learning_rate)

print('\n', w, '\n', b)

# w = 0
# b = 0
# for i in range(len(train_data[feature])):
#     w, b = singleBatchTrain(w, b, train_data[feature].values, train_data[label], 0.01)
#     print(w, '\n', b, '\n')
