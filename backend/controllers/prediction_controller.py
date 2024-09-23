# app/controllers/prediction_controller.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model
from cache import get_cached_data, set_cached_data
from utils.logger import log_model_usage, get_model_metrics, update_model_metrics


class PredictionController:
    def __init__(self):
        self.lstm_model = load_model("lstm_model.h5")
        self.nn_model = load_model("neural_network_model.h5")
        self.lstm_scaler = MinMaxScaler(feature_range=(0, 1))
        self.nn_scaler = StandardScaler()

    def get_avax_data(self):
        cached_data = get_cached_data("avax_data")
        if cached_data is not None:
            return cached_data

        avax = yf.Ticker("AVAX-USD")
        avax_data = avax.history(period="5y")
        set_cached_data("avax_data", avax_data)
        return avax_data

    def predict_lstm(self, days=30):
        log_model_usage("LSTM", "1.0")
        avax_data = self.get_avax_data()

        scaled_data = self.lstm_scaler.fit_transform(avax_data[["Close"]])

        sequence_length = 60
        X = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length : i, 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        predictions = self._predict_lstm_future(X, days)

        future_dates = [
            (avax_data.index[-1] + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, days + 1)
        ]

        # Get LSTM metrics
        lstm_metrics = get_model_metrics("LSTM")

        return {
            "dates": future_dates,
            "predictions": predictions.flatten().tolist(),
            "metrics": lstm_metrics,
        }

    def _predict_lstm_future(self, X_test, days):
        last_sequence = X_test[-1]
        predictions = []
        sequence_length = X_test.shape[1]

        for _ in range(days):
            pred = self.lstm_model.predict(last_sequence.reshape(1, sequence_length, 1))
            predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)[-sequence_length:]
            last_sequence = last_sequence.reshape(sequence_length, 1)

        predictions = self.lstm_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        return predictions

    def predict_rsi(self, days=30):
        log_model_usage("RSI Neural Network", "1.0")
        avax_data = self.get_avax_data()

        avax_data["RSI"] = self._calculate_rsi(avax_data)
        avax_data["SMA_30"] = avax_data["Close"].rolling(window=30).mean()
        avax_data["SMA_100"] = avax_data["Close"].rolling(window=100).mean()

        avax_data.dropna(inplace=True)

        features = avax_data[["RSI", "SMA_30", "SMA_100"]]
        features_scaled = self.nn_scaler.fit_transform(features)

        predictions = self._predict_with_neural_network(features_scaled, days)

        future_dates = [
            (avax_data.index[-1] + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, days + 1)
        ]

        # Get RSI metrics
        rsi_metrics = get_model_metrics("RSI Neural Network")

        return {
            "dates": future_dates,
            "predictions": predictions.tolist(),
            "latest_rsi": avax_data["RSI"].iloc[-1],
            "recommendation": self._rsi_recommendation(avax_data["RSI"].iloc[-1]),
            "metrics": rsi_metrics,
        }

    def _calculate_rsi(self, data, window=14):
        delta = data["Close"].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _predict_with_neural_network(self, data, days=30):
        X_test = np.array(data[-days:])
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        predictions = self.nn_model.predict(X_test)
        return predictions.flatten()

    def _rsi_recommendation(self, rsi_value):
        if rsi_value > 70:
            return "Overbought: Consider Selling"
        elif rsi_value < 30:
            return "Oversold: Consider Buying"
        else:
            return "Hold: No Strong Signal"


def initialize_metrics():
    update_model_metrics("LSTM", {"MSE": 12.34, "RMSE": 3.51, "MAE": 2.76})
    update_model_metrics("RSI Neural Network", {"accuracy": 0.4540, "loss": 0.9933})
