from tkinter import messagebox
from tkinter import *
from tkinter import messagebox
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import os
import json

MPLCONFIGDIR = os.path.join(os.path.dirname(__file__), ".matplotlib")
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.makedirs(MPLCONFIGDIR, exist_ok=True)

import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR  #SVR regression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor #random forest regression class

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error,r2_score

main = Tk()
main.title("Discovery and Prediction of Stock Index Pattern via Three-Stage Architecture of TICC, TPA-LSTM and Multivariate LSTM-FCNs")
main.geometry("1300x1200")


global filename
global X_train, X_test, y_train, y_test
global X_train_lstm, X_test_lstm
global X_train_ml,X_test_ml
global x_scaler, y_scaler

rmse_scores = {}
MODEL_VERSION = "tpa_lstm_v4_calibrated_return"
MODEL_PATH = os.path.join("model", "tpa_lstm_model.keras")
MODEL_META_PATH = os.path.join("model", "tpa_lstm_model_meta.json")
MLSTM_FCN_MODEL_VERSION = "multivariate_lstm_fcn_v1_calibrated_return"
MLSTM_FCN_MODEL_PATH = os.path.join("model", "multivariate_lstm_fcn_model.h5")
MLSTM_FCN_MODEL_META_PATH = os.path.join("model", "multivariate_lstm_fcn_model_meta.json")
WINDOW_SIZE = 30
FEATURE_COLUMNS = [
    'value',
    'MA_5',
    'MA_10',
    'MA_20',
    'MA_30',
    'MA_50',
    'MA_100',
    'EMA_5',
    'EMA_10',
    'EMA_20',
    'EMA_50',
    'Return',
    'Return_Lag_1',
    'Return_Lag_2',
    'Return_Lag_3',
    'Return_Lag_5',
    'Return_Lag_10',
    'Volatility_5',
    'Volatility_10',
    'Volatility_20',
    'Momentum_1',
    'Momentum_3',
    'Momentum_5',
    'Momentum_10',
    'Momentum_20',
    'Price_MA5_Ratio',
    'Price_MA20_Ratio',
    'Price_MA50_Ratio',
    'MA5_MA20_Ratio',
    'MA20_MA50_Ratio'
]

def clear_output():
    text.delete('1.0', END)
    main.update_idletasks()


def append_output(message):
    text.insert(END, message)
    text.see(END)
    main.update_idletasks()


def start_plot(figsize, dpi=None):
    plt.close('all')
    return plt.figure(figsize=figsize, dpi=dpi)


def show_plot():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    main.update_idletasks()


def set_buttons_state(state):
    for button_name in [
        'uploadButton',
        'preprocessButton',
        'svmButton',
        'rfButton',
        'baselineButton',
        'xgbButton',
        'nbButton',
        'lstmButton',
        'mlstmFcnButton',
        'raeButton',
        'forecastButton',
        'exitButton'
    ]:
        button = globals().get(button_name)
        if button is not None:
            button.config(state=state)
    main.update_idletasks()


def run_with_busy(action):
    set_buttons_state(DISABLED)
    try:
        action()
    finally:
        set_buttons_state(NORMAL)


def uploadDataset():
    global filename
    global dataset
    clear_output()
    filename = askopenfilename(initialdir = "Dataset")
    tf1.delete(0, END)
    tf1.insert(END, str(filename))
    append_output("Dataset Loaded\n\n")
    dataset = pd.read_csv(filename)
    append_output(str(dataset.head()))
    start_plot(figsize=(16,10), dpi=100)
    plt.plot(dataset.date[0:10], dataset.value[0:10], color='tab:red')
    plt.gca().set(title="Hang-Send Stock Daily Closing Prices", xlabel='Date', ylabel="Closing Price")
    show_plot()
    
def preprocessDataset():
    global X_train, X_test, y_train, y_test
    global X_train_lstm, X_test_lstm
    global X_train_ml, X_test_ml
    global x_scaler, y_scaler, lstm_y_scaler
    global y_train_lstm_target, y_test_lstm_target
    global y_train_actual, y_test_actual
    global train_previous_close, test_previous_close
    global dataset

    clear_output()

    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset.sort_values('date', inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    dataset['MA_5'] = dataset['value'].rolling(window=5).mean()
    dataset['MA_10'] = dataset['value'].rolling(window=10).mean()
    dataset['MA_20'] = dataset['value'].rolling(window=20).mean()
    dataset['MA_30'] = dataset['value'].rolling(window=30).mean()
    dataset['MA_50'] = dataset['value'].rolling(window=50).mean()
    dataset['MA_100'] = dataset['value'].rolling(window=100).mean()
    dataset['EMA_5'] = dataset['value'].ewm(span=5, adjust=False).mean()
    dataset['EMA_10'] = dataset['value'].ewm(span=10, adjust=False).mean()
    dataset['EMA_20'] = dataset['value'].ewm(span=20, adjust=False).mean()
    dataset['EMA_50'] = dataset['value'].ewm(span=50, adjust=False).mean()
  
    dataset['Return'] = dataset['value'].pct_change()
    dataset['Return_Lag_1'] = dataset['Return'].shift(1)
    dataset['Return_Lag_2'] = dataset['Return'].shift(2)
    dataset['Return_Lag_3'] = dataset['Return'].shift(3)
    dataset['Return_Lag_5'] = dataset['Return'].shift(5)
    dataset['Return_Lag_10'] = dataset['Return'].shift(10)
    dataset['Volatility_5'] = dataset['Return'].rolling(window=5).std()
    dataset['Volatility_10'] = dataset['Return'].rolling(window=10).std()
    dataset['Volatility_20'] = dataset['Return'].rolling(window=20).std()
    dataset['Momentum_1'] = dataset['value'] - dataset['value'].shift(1)
    dataset['Momentum_3'] = dataset['value'] - dataset['value'].shift(3)
    dataset['Momentum_5'] = dataset['value'] - dataset['value'].shift(5)
    dataset['Momentum_10'] = dataset['value'] - dataset['value'].shift(10)
    dataset['Momentum_20'] = dataset['value'] - dataset['value'].shift(20)
    dataset['Price_MA5_Ratio'] = dataset['value'] / dataset['MA_5']
    dataset['Price_MA20_Ratio'] = dataset['value'] / dataset['MA_20']
    dataset['Price_MA50_Ratio'] = dataset['value'] / dataset['MA_50']
    dataset['MA5_MA20_Ratio'] = dataset['MA_5'] / dataset['MA_20']
    dataset['MA20_MA50_Ratio'] = dataset['MA_20'] / dataset['MA_50']
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)


    values = dataset[FEATURE_COLUMNS].values

    X = []
    Y = []

    for i in range(WINDOW_SIZE, len(values)):
        X.append(values[i-WINDOW_SIZE:i])
        Y.append(values[i, 0])

    X = np.array(X)
    Y = np.array(Y)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = Y[:split]
    y_test = Y[split:]
    y_train_actual = y_train.reshape(-1, 1)
    y_test_actual = y_test.reshape(-1, 1)
    train_previous_close = X_train[:, -1, 0].reshape(-1, 1)
    test_previous_close = X_test[:, -1, 0].reshape(-1, 1)
    y_train_return = np.log(y_train_actual / train_previous_close)
    y_test_return = np.log(y_test_actual / test_previous_close)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    lstm_y_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1,X_train.shape[2])
    X_test_2d = X_test.reshape(-1,X_test.shape[2]) 

    X_train_2d = x_scaler.fit_transform(X_train_2d)
    X_test_2d = x_scaler.transform(X_test_2d)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))
    y_train_lstm_target = lstm_y_scaler.fit_transform(y_train_return)
    y_test_lstm_target = lstm_y_scaler.transform(y_test_return)

    X_train_lstm = X_train_2d.reshape(
        X_train.shape
    )

    X_test_lstm = X_test_2d.reshape(
        X_test.shape
    )
    X_train_ml = X_train_lstm.reshape(
    X_train_lstm.shape[0],
    -1
     )

    X_test_ml = X_test_lstm.reshape(
    X_test_lstm.shape[0],
    -1
     )

    append_output(f"Training samples: {len(X_train)}\n")
    append_output(f"Testing samples: {len(X_test)}\n")
    append_output(f"Window size: {WINDOW_SIZE} days\n")
    append_output(f"Features: {len(FEATURE_COLUMNS)}\n")
    append_output("TPA-LSTM target: next-day log return\n")

def evaluate_model(model_name, actual, predictions):

    global rmse_scores

    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)

    rmse_scores[model_name] = rmse

    append_output(f"\n{model_name} Results\n")
    append_output("-" * 40 + "\n")

    append_output(f"MAE  : {mae:.4f}\n")
    append_output(f"RMSE : {rmse:.4f}\n")
    append_output(f"R2   : {r2:.4f}\n\n")


def saved_model_matches(expected_shape):

    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_META_PATH):
        return False

    try:
        with open(MODEL_META_PATH, "r") as meta_file:
            metadata = json.load(meta_file)
    except (OSError, ValueError):
        return False

    return (
        metadata.get("version") == MODEL_VERSION and
        metadata.get("window_size") == WINDOW_SIZE and
        metadata.get("feature_columns") == FEATURE_COLUMNS and
        metadata.get("target") == "next_day_log_return" and
        tuple(metadata.get("input_shape", [])) == expected_shape
    )


def save_model_metadata(expected_shape):

    metadata = {
        "version": MODEL_VERSION,
        "window_size": WINDOW_SIZE,
        "feature_columns": FEATURE_COLUMNS,
        "target": "next_day_log_return",
        "input_shape": list(expected_shape)
    }

    with open(MODEL_META_PATH, "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)


def saved_mlstm_fcn_model_matches(expected_shape):

    if not os.path.exists(MLSTM_FCN_MODEL_PATH) or not os.path.exists(MLSTM_FCN_MODEL_META_PATH):
        return False

    try:
        with open(MLSTM_FCN_MODEL_META_PATH, "r") as meta_file:
            metadata = json.load(meta_file)
    except (OSError, ValueError):
        return False

    return (
        metadata.get("version") == MLSTM_FCN_MODEL_VERSION and
        metadata.get("window_size") == WINDOW_SIZE and
        metadata.get("feature_columns") == FEATURE_COLUMNS and
        metadata.get("target") == "next_day_log_return" and
        tuple(metadata.get("input_shape", [])) == expected_shape
    )


def save_mlstm_fcn_model_metadata(expected_shape):

    metadata = {
        "version": MLSTM_FCN_MODEL_VERSION,
        "window_size": WINDOW_SIZE,
        "feature_columns": FEATURE_COLUMNS,
        "target": "next_day_log_return",
        "input_shape": list(expected_shape)
    }

    with open(MLSTM_FCN_MODEL_META_PATH, "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)


def calibrate_lstm_blend(model):

    global lstm_blend_alpha

    val_split = int(len(X_train_lstm) * 0.85)
    X_val = X_train_lstm[val_split:]
    previous_close = X_train[val_split:, -1, 0].reshape(-1, 1)
    actual = y_train_actual[val_split:]

    predicted_returns = model.predict(X_val, verbose=0)
    predicted_returns = lstm_y_scaler.inverse_transform(predicted_returns)
    lstm_predictions = previous_close * np.exp(predicted_returns)

    best_alpha = 1.0
    best_rmse = float("inf")

    for alpha in np.linspace(0.0, 1.0, 21):
        blended = (alpha * lstm_predictions) + ((1.0 - alpha) * previous_close)
        rmse = np.sqrt(mean_squared_error(actual, blended))

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    lstm_blend_alpha = best_alpha
    return best_alpha, best_rmse


def runSVM():

    clear_output()
    append_output("Running SVM model. Please wait...\n")

    model = SVR(
        C=50,
        epsilon=0.01,
        gamma='scale'
    )

    model.fit(X_train_ml, y_train.ravel())

    predictions = model.predict(X_test_ml)

    predictions = y_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    )

    actual = y_scaler.inverse_transform(y_test)

    evaluate_model("SVM", actual, predictions)

    start_plot(figsize=(12, 6))

    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')

    plt.title('SVM Stock Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')

    plt.legend()
    plt.grid(alpha=0.3)

    show_plot()


def runRandomForest():

    clear_output()
    append_output("Running Random Forest model. Please wait...\n")

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_ml, y_train.ravel())

    predictions = model.predict(X_test_ml)

    predictions = y_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    )

    actual = y_scaler.inverse_transform(y_test)

    evaluate_model("Random Forest", actual, predictions)

    start_plot(figsize=(12, 6))

    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')

    plt.title('Random Forest Stock Prediction')

    plt.legend()
    plt.grid(alpha=0.3)

    show_plot()


def runNaiveBaseline():

    clear_output()
    append_output("Running Previous Close Baseline. Please wait...\n")
    append_output(
        "This baseline predicts tomorrow's price as today's closing price. "
        "Stock-index models must beat this to be useful.\n"
    )

    actual = y_test_actual
    predictions = test_previous_close

    evaluate_model("Previous Close Baseline", actual, predictions)

    start_plot(figsize=(12, 6))

    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Previous Close Baseline')

    plt.title('Previous Close Baseline Stock Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')

    plt.legend()
    plt.grid(alpha=0.3)

    show_plot()


def runXGBoost():

    global X_train_ml, X_test_ml
    global y_train_lstm_target, y_test_lstm_target
    global y_train_actual, y_test_actual, lstm_y_scaler

    clear_output()
    append_output("Running XGBoost Regressor model. Please wait...\n")

    try:
        from xgboost import XGBRegressor
    except ImportError:
        append_output(
            "XGBoost is not installed.\n"
            "Install it with: pip install xgboost==1.6.2\n"
            "Or run: pip install -r requirements.txt\n"
        )
        messagebox.showerror(
            "Missing dependency",
            "XGBoost is not installed. Please install xgboost==1.6.2 and run again."
        )
        return

    model = XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1
    )

    val_split = int(len(X_train_ml) * 0.85)
    X_fit = X_train_ml[:val_split]
    y_fit = y_train_lstm_target[:val_split].ravel()
    X_val = X_train_ml[val_split:]
    previous_close_val = X_train[val_split:, -1, 0].reshape(-1, 1)
    actual_val = y_train_actual[val_split:]

    model.fit(X_fit, y_fit)

    val_returns = model.predict(X_val).reshape(-1, 1)
    val_returns = lstm_y_scaler.inverse_transform(val_returns)
    val_predictions = previous_close_val * np.exp(val_returns)

    best_alpha = 1.0
    best_rmse = float("inf")

    for alpha in np.linspace(0.0, 1.0, 21):
        blended = (alpha * val_predictions) + ((1.0 - alpha) * previous_close_val)
        rmse = np.sqrt(mean_squared_error(actual_val, blended))

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    predicted_returns = model.predict(X_test_ml).reshape(-1, 1)
    predicted_returns = lstm_y_scaler.inverse_transform(predicted_returns)

    previous_close = X_test[:, -1, 0].reshape(-1, 1)
    xgb_predictions = previous_close * np.exp(predicted_returns)
    predictions = (best_alpha * xgb_predictions) + ((1.0 - best_alpha) * previous_close)
    actual = y_test_actual

    append_output(f"Validation blend alpha: {best_alpha:.2f} (RMSE {best_rmse:.4f})\n")
    if best_alpha == 0.0:
        append_output(
            "Calibration selected the previous-close baseline. "
            "The raw XGBoost signal did not improve validation RMSE.\n"
        )

    evaluate_model("XGBoost Raw Regressor", actual, xgb_predictions)
    evaluate_model("XGBoost Calibrated", actual, predictions)

    append_output("Sample Predictions\n")
    append_output("-" * 50 + "\n")

    for i in range(min(20, len(actual))):
        append_output(
            f"Actual: {actual[i][0]:.2f}    "
            f"Predicted: {predictions[i][0]:.2f}\n"
        )

    start_plot(figsize=(12, 6))

    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')

    plt.title('XGBoost Regressor Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')

    plt.legend()
    plt.grid(alpha=0.3)

    show_plot()
    
def runBayesianRidge():

    clear_output()
    append_output("Running Bayesian Ridge model. Please wait...\n")

    model = BayesianRidge()

    model.fit(X_train_ml, y_train.ravel())

    predictions = model.predict(X_test_ml)

    predictions = y_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    )

    actual = y_scaler.inverse_transform(y_test)

    evaluate_model("Bayesian Ridge", actual, predictions)

    start_plot(figsize=(12, 6))

    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')

    plt.title('Bayesian Ridge Stock Prediction')

    plt.legend()
    plt.grid(alpha=0.3)

    show_plot()

def runTPALSTM():
    global X_train_lstm, X_test_lstm
    global y_train_lstm_target, y_test_lstm_target
    global y_train_actual, y_test_actual, lstm_y_scaler

    clear_output()
    append_output("Preparing TPA-LSTM model. Please wait...\n")

    os.makedirs("model", exist_ok=True)

    expected_shape = (
        X_train_lstm.shape[1],
        X_train_lstm.shape[2]
    )

    def create_tpa_lstm_model():
        new_model = Sequential()

        new_model.add(
            LSTM(
                units=96,
                return_sequences=True,
                input_shape=expected_shape
            )
        )

        new_model.add(Dropout(0.15))

        new_model.add(LSTM(units=48))

        new_model.add(Dropout(0.15))

        new_model.add(Dense(24, activation='relu'))

        new_model.add(Dense(1))

        new_model.compile(
            optimizer=Adam(lr=0.001),
            loss='mean_squared_error'
        )

        return new_model

    model = None

    if saved_model_matches(expected_shape):
        saved_model = load_model(MODEL_PATH)
        saved_shape = saved_model.input_shape[1:]

        if saved_shape == expected_shape:
            model = saved_model
            append_output("Loaded saved model\n\n")
    elif os.path.exists(MODEL_PATH):
        append_output("Saved model is from an older training setup. Retraining improved model.\n\n")

    if model is None:
        model = create_tpa_lstm_model()
        append_output("Training TPA-LSTM. This may take a few minutes...\n")

        val_split = int(len(X_train_lstm) * 0.85)
        X_fit = X_train_lstm[:val_split]
        y_fit = y_train_lstm_target[:val_split]
        X_val = X_train_lstm[val_split:]
        y_val = y_train_lstm_target[val_split:]

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        history = model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        model.save(MODEL_PATH)
        save_model_metadata(expected_shape)

        append_output(f"Training completed in {len(history.history['loss'])} epochs\n\n")

    blend_alpha, blend_rmse = calibrate_lstm_blend(model)
    append_output(f"Validation blend alpha: {blend_alpha:.2f} (RMSE {blend_rmse:.4f})\n")
    append_output("Generating TPA-LSTM predictions...\n")
    predicted_returns = model.predict(X_test_lstm, verbose=0)
    predicted_returns = lstm_y_scaler.inverse_transform(predicted_returns)

    previous_close = X_test[:, -1, 0].reshape(-1, 1)
    lstm_predictions = previous_close * np.exp(predicted_returns)
    predictions = (blend_alpha * lstm_predictions) + ((1.0 - blend_alpha) * previous_close)
    actual = y_test_actual

    evaluate_model(
        "TPA-LSTM",
        actual,
        predictions
    )

    append_output("Sample Predictions\n")
    append_output("-" * 50 + "\n")

    for i in range(min(20, len(actual))):
        append_output(
            f"Actual: {actual[i][0]:.2f}    "
            f"Predicted: {predictions[i][0]:.2f}\n"
        )

    start_plot(figsize=(12, 6))

    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')

    plt.title('TPA-LSTM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')

    plt.legend()
    plt.grid(alpha=0.3)

    show_plot()


def runMultivariateLSTMFCN():
    global X_train_lstm, X_test_lstm
    global y_train_lstm_target, y_test_lstm_target
    global y_train_actual, y_test_actual, lstm_y_scaler

    clear_output()
    append_output("Preparing Multivariate LSTM-FCN model. Please wait...\n")

    os.makedirs("model", exist_ok=True)

    expected_shape = (
        X_train_lstm.shape[1],
        X_train_lstm.shape[2]
    )

    def create_multivariate_lstm_fcn_model():
        input_layer = Input(shape=expected_shape)

        lstm_branch = LSTM(64)(input_layer)
        lstm_branch = Dropout(0.2)(lstm_branch)

        conv_branch = Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv_branch = BatchNormalization()(conv_branch)
        conv_branch = Activation('relu')(conv_branch)

        conv_branch = Conv1D(filters=256, kernel_size=5, padding='same')(conv_branch)
        conv_branch = BatchNormalization()(conv_branch)
        conv_branch = Activation('relu')(conv_branch)

        conv_branch = Conv1D(filters=128, kernel_size=3, padding='same')(conv_branch)
        conv_branch = BatchNormalization()(conv_branch)
        conv_branch = Activation('relu')(conv_branch)
        conv_branch = GlobalAveragePooling1D()(conv_branch)

        merged = concatenate([lstm_branch, conv_branch])
        merged = Dropout(0.2)(merged)
        output_layer = Dense(1)(merged)

        new_model = Model(inputs=input_layer, outputs=output_layer)
        new_model.compile(
            optimizer=Adam(lr=0.001),
            loss='mean_squared_error'
        )

        return new_model

    model = None

    if saved_mlstm_fcn_model_matches(expected_shape):
        saved_model = load_model(MLSTM_FCN_MODEL_PATH)
        saved_shape = saved_model.input_shape[1:]

        if saved_shape == expected_shape:
            model = saved_model
            append_output("Loaded saved Multivariate LSTM-FCN model\n\n")
    elif os.path.exists(MLSTM_FCN_MODEL_PATH):
        append_output("Saved Multivariate LSTM-FCN model is from an older training setup. Retraining model.\n\n")

    if model is None:
        model = create_multivariate_lstm_fcn_model()
        append_output("Training Multivariate LSTM-FCN. This may take a few minutes...\n")

        val_split = int(len(X_train_lstm) * 0.85)
        X_fit = X_train_lstm[:val_split]
        y_fit = y_train_lstm_target[:val_split]
        X_val = X_train_lstm[val_split:]
        y_val = y_train_lstm_target[val_split:]

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        history = model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        model.save(MLSTM_FCN_MODEL_PATH)
        save_mlstm_fcn_model_metadata(expected_shape)

        append_output(f"Training completed in {len(history.history['loss'])} epochs\n\n")

    blend_alpha, blend_rmse = calibrate_lstm_blend(model)
    append_output(f"Validation blend alpha: {blend_alpha:.2f} (RMSE {blend_rmse:.4f})\n")
    append_output("Generating Multivariate LSTM-FCN predictions...\n")

    predicted_returns = model.predict(X_test_lstm, verbose=0)
    predicted_returns = lstm_y_scaler.inverse_transform(predicted_returns)

    previous_close = X_test[:, -1, 0].reshape(-1, 1)
    raw_predictions = previous_close * np.exp(predicted_returns)
    predictions = (blend_alpha * raw_predictions) + ((1.0 - blend_alpha) * previous_close)
    actual = y_test_actual

    evaluate_model(
        "Multivariate LSTM-FCN",
        actual,
        predictions
    )

    append_output("Sample Predictions\n")
    append_output("-" * 50 + "\n")

    for i in range(min(20, len(actual))):
        append_output(
            f"Actual: {actual[i][0]:.2f}    "
            f"Predicted: {predictions[i][0]:.2f}\n"
        )

    start_plot(figsize=(12, 6))

    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')

    plt.title('Multivariate LSTM-FCN Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')

    plt.legend()
    plt.grid(alpha=0.3)

    show_plot()


def build_feature_row_from_prices(prices):

    close_series = pd.Series(prices)
    return_series = close_series.pct_change()
    ma_5 = close_series.rolling(window=5).mean().iloc[-1]
    ma_20 = close_series.rolling(window=20).mean().iloc[-1]
    ma_50 = close_series.rolling(window=50).mean().iloc[-1]

    feature_row = {
        'value': close_series.iloc[-1],
        'MA_5': ma_5,
        'MA_10': close_series.rolling(window=10).mean().iloc[-1],
        'MA_20': ma_20,
        'MA_30': close_series.rolling(window=30).mean().iloc[-1],
        'MA_50': ma_50,
        'MA_100': close_series.rolling(window=100).mean().iloc[-1],
        'EMA_5': close_series.ewm(span=5, adjust=False).mean().iloc[-1],
        'EMA_10': close_series.ewm(span=10, adjust=False).mean().iloc[-1],
        'EMA_20': close_series.ewm(span=20, adjust=False).mean().iloc[-1],
        'EMA_50': close_series.ewm(span=50, adjust=False).mean().iloc[-1],
        'Return': return_series.iloc[-1],
        'Return_Lag_1': return_series.shift(1).iloc[-1],
        'Return_Lag_2': return_series.shift(2).iloc[-1],
        'Return_Lag_3': return_series.shift(3).iloc[-1],
        'Return_Lag_5': return_series.shift(5).iloc[-1],
        'Return_Lag_10': return_series.shift(10).iloc[-1],
        'Volatility_5': return_series.rolling(window=5).std().iloc[-1],
        'Volatility_10': return_series.rolling(window=10).std().iloc[-1],
        'Volatility_20': return_series.rolling(window=20).std().iloc[-1],
        'Momentum_1': close_series.iloc[-1] - close_series.iloc[-2],
        'Momentum_3': close_series.iloc[-1] - close_series.iloc[-4],
        'Momentum_5': close_series.iloc[-1] - close_series.iloc[-6],
        'Momentum_10': close_series.iloc[-1] - close_series.iloc[-11],
        'Momentum_20': close_series.iloc[-1] - close_series.iloc[-21],
        'Price_MA5_Ratio': close_series.iloc[-1] / ma_5,
        'Price_MA20_Ratio': close_series.iloc[-1] / ma_20,
        'Price_MA50_Ratio': close_series.iloc[-1] / ma_50,
        'MA5_MA20_Ratio': ma_5 / ma_20,
        'MA20_MA50_Ratio': ma_20 / ma_50
    }

    return np.array([[feature_row[column] for column in FEATURE_COLUMNS]])


def forecast_next_days(days=7):

    global X_test_lstm, lstm_y_scaler, x_scaler, dataset

    if 'X_test_lstm' not in globals() or 'lstm_y_scaler' not in globals() or 'x_scaler' not in globals():
        messagebox.showerror(
            "Error",
            "Please upload and preprocess the dataset before forecasting"
        )
        return None

    if not os.path.exists(MODEL_PATH):
        messagebox.showerror(
            "Error",
            "Please run the TPA-LSTM model before forecasting"
        )
        return None

    expected_shape = (
        X_test_lstm.shape[1],
        X_test_lstm.shape[2]
    )

    if not saved_model_matches(expected_shape):
        messagebox.showerror(
            "Error",
            "Saved TPA-LSTM model does not match the current training setup. Please run the TPA-LSTM model again."
        )
        return None

    model = load_model(MODEL_PATH)
    blend_alpha, _ = calibrate_lstm_blend(model)

    current_window = X_test_lstm[-1].copy()
    close_history = dataset['value'].astype(float).tolist()

    predictions = []

    for _ in range(days):

        pred = model.predict(
            current_window.reshape(
                1,
                current_window.shape[0],
                current_window.shape[1]
            ),
            verbose=0
        )

        pred_return = lstm_y_scaler.inverse_transform(pred)[0][0]
        lstm_value = close_history[-1] * np.exp(pred_return)
        pred_value = (blend_alpha * lstm_value) + ((1.0 - blend_alpha) * close_history[-1])
        predictions.append(pred_value)
        close_history.append(pred_value)

        next_row = x_scaler.transform(
            build_feature_row_from_prices(close_history)
        )[0]

        current_window = np.vstack(
            [current_window[1:], next_row]
        )

    return np.array(predictions).reshape(-1, 1)

def plot_forecast_next_days(days=7):

    future = forecast_next_days(days)

    if future is None:
        return

    start_plot(figsize=(10, 6))

    plt.plot(
        range(1, days + 1),
        future,
        marker='o'
    )

    plt.title(f"Next {days}-Day Forecast")

    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Price")

    plt.grid(alpha=0.3)

    show_plot()

def graph():

    if len(rmse_scores) == 0:
        messagebox.showerror(
            "Error",
            "Run at least one model first"
        )
        return

    names = list(rmse_scores.keys())
    values = list(rmse_scores.values())

    start_plot(figsize=(10, 6))

    plt.bar(names, values)

    plt.title("RMSE Comparison")
    plt.ylabel("RMSE")

    plt.xticks(rotation=15)

    plt.grid(axis='y', alpha=0.3)

    show_plot()

def close():
    main.destroy()


font = ('times', 15, 'bold')
title = Label(main, text='Discovery and Prediction of Stock Index Pattern via Three-Stage Architecture of TICC, TPA-LSTM and Multivariate LSTM-FCNs')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = Button(main, text="Upload Hang-Send Stock Dataset", command=lambda: run_with_busy(uploadDataset), bg='#ffb3fe')
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=lambda: run_with_busy(preprocessDataset), bg='#ffb3fe')
preprocessButton.place(x=470,y=150)
preprocessButton.config(font=font1)

svmButton = Button(main,text="Run SVM Algorithm", command=lambda: run_with_busy(runSVM), bg='#ffb3fe')
svmButton.place(x=790,y=150)
svmButton.config(font=font1)

baselineButton = Button(main,text="Run Previous Close Baseline", command=lambda: run_with_busy(runNaiveBaseline), bg='#ffb3fe')
baselineButton.place(x=1050,y=150)
baselineButton.config(font=font1)

rfButton = Button(main,text="Run Random Forest Algorithm", command=lambda: run_with_busy(runRandomForest), bg='#ffb3fe')
rfButton.place(x=50,y=200)
rfButton.config(font=font1)

nbButton = Button(main,text="Run Bayesian Ridge", command=lambda: run_with_busy(runBayesianRidge), bg='#ffb3fe')
nbButton.place(x=470,y=200)
nbButton.config(font=font1)

lstmButton = Button(main,text="Run Propose TPA-LSTM", command=lambda: run_with_busy(runTPALSTM), bg='#ffb3fe')
lstmButton.place(x=790,y=200)
lstmButton.config(font=font1)

xgbButton = Button(main,text="Run XGBoost Regressor", command=lambda: run_with_busy(runXGBoost), bg='#ffb3fe')
xgbButton.place(x=1050,y=200)
xgbButton.config(font=font1)

mlstmFcnButton = Button(main,text="Run Multivariate LSTM-FCN", command=lambda: run_with_busy(runMultivariateLSTMFCN), bg='#ffb3fe')
mlstmFcnButton.place(x=790,y=250)
mlstmFcnButton.config(font=font1)

raeButton = Button(main,text="Relative Absolute Error Graph",command=lambda: run_with_busy(graph), bg='#ffb3fe')
raeButton.place(x=50,y=250)
raeButton.config(font=font1)

forecastButton = Button(
    main,
    text="Forecast Next 7 Days",
    command=lambda: run_with_busy(lambda: plot_forecast_next_days(7)),
    bg='#ffb3fe'
)

forecastButton.place(x=470, y=250)
forecastButton.config(font=font1)

exitButton = Button(main,text="Exit", command=lambda: run_with_busy(close), bg='#ffb3fe')
exitButton.place(x=1100,y=250)
exitButton.config(font=font1)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
