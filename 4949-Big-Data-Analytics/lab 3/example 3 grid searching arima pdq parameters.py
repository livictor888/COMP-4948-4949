from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

PATH = "C:\\datasets\\"
series = read_csv(PATH + 'daily-min-temperatures.csv', header=0, index_col=0)


# Evaluate an ARIMA model for a given order (p,d,q).
def evaluate_arima_model(X, arima_order):
    # Prepare training dataset.
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]

    # Make predictions.
    predictions = list()

    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    # Calculate out of sample error,
    error = mean_squared_error(test, predictions)
    return error


# Evaluate combinations of p, d and q values for an ARIMA model.
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# Set parameter ranges.
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)

# Evaluate performance.
evaluate_models(series.values, p_values, d_values, q_values)
