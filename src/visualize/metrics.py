from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def get_metrics(y_train, predictions_train, y_test, predictions_test):
    r2_train = r2_score(y_train, predictions_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))


    r2_test = r2_score(y_test, predictions_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
    metrics = {
        "train": {"r2": r2_train, "rmse": rmse_train},
        "test": {"r2": r2_test, "rmse": rmse_test},
    }
    return metrics