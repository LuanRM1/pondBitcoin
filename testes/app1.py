from fastapi import FastAPI, Query
from typing import Optional
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model

app = FastAPI()

rsi_model = load_model("neural_network_model.h5")
rsi_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

lstm_model = load_model("lstm_model.h5")
lstm_model.compile(optimizer="adam", loss="mean_squared_error")


def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_data():
    avax = yf.Ticker("AVAX-USD")
    avax_data = avax.history(period="5y")

    # Calcular indicadores técnicos
    avax_data["RSI"] = calculate_rsi(avax_data["Close"])
    avax_data["SMA_30"] = avax_data["Close"].rolling(window=30).mean()
    avax_data["SMA_100"] = avax_data["Close"].rolling(window=100).mean()

    # Eliminar valores NaN criados durante o cálculo dos indicadores
    avax_data.dropna(inplace=True)

    # Selecionar as features (RSI, SMA_30, SMA_100)
    features = avax_data[["RSI", "SMA_30", "SMA_100"]]
    target = (avax_data["Close"].shift(-1) > avax_data["Close"]).astype(
        int
    )  # Prever direção de alta/baixa

    # Normalizar as features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return avax_data, scaled_features, target, scaler


# Função para carregar e processar dados para o modelo LSTM
def get_lstm_data():
    avax = yf.Ticker("AVAX-USD")
    avax_data = avax.history(period="5y")

    data = avax_data[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return data, scaled_data, scaler


# Função para gerar previsões futuras com o modelo RSI
def predict_rsi_future(scaled_features, days=30):
    future_predictions = rsi_model.predict(scaled_features[-days:])
    return (future_predictions > 0.5).astype(int)  # Retornar 1 (compra) ou 0 (venda)


# Função para gerar previsões futuras com o modelo LSTM
def predict_lstm_future(scaler, X_test, days=30):
    last_sequence = X_test[-1]
    predictions = []
    sequence_length = X_test.shape[1]

    for _ in range(days):
        pred = lstm_model.predict(last_sequence.reshape(1, sequence_length, 1))
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)[-sequence_length:]
        last_sequence = last_sequence.reshape(sequence_length, 1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


# Erros e acurácia pré-calculados para LSTM e RSI
LSTM_ERRORS = {"mse": 12.34, "rmse": 3.51, "mae": 2.76}

RSI_ACCURACY = {"accuracy": 0.46}


# Endpoint para previsões usando o modelo RSI
@app.get("/rsi-predictions/")
async def get_rsi_predictions(
    days: Optional[int] = Query(30, description="Número de dias para prever")
):
    avax_data, scaled_features, target, scaler = get_data()

    # Previsões para o número de dias solicitado
    future_predictions = predict_rsi_future(scaled_features, days)

    # Datas para o gráfico
    future_dates = (
        pd.date_range(start=avax_data.index[-1], periods=days + 1, closed="right")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    return {
        "historical_dates": avax_data.index.strftime("%Y-%m-%d").tolist(),
        "historical_prices": avax_data["Close"].tolist(),
        "future_dates": future_dates,
        "future_predictions": future_predictions.flatten().tolist(),
    }


# Endpoint para previsões usando o modelo LSTM
@app.get("/lstm-predictions/")
async def get_lstm_predictions(
    days: Optional[int] = Query(30, description="Número de dias para prever")
):
    data, scaled_data, scaler = get_lstm_data()

    # Previsão futura
    future_predictions = predict_lstm_future(scaler, scaled_data, days)

    # Datas para o gráfico
    future_dates = (
        pd.date_range(start=data.index[-1], periods=days + 1, closed="right")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    return {
        "historical_dates": data.index.strftime("%Y-%m-%d").tolist(),
        "historical_prices": data["Close"].tolist(),
        "future_dates": future_dates,
        "future_predictions": future_predictions.flatten().tolist(),
    }


# Endpoint para exibir erros pré-calculados do modelo LSTM
@app.get("/lstm-errors/")
async def get_lstm_errors():
    return {
        "LSTM Mean Squared Error (MSE)": LSTM_ERRORS["mse"],
        "Root Mean Squared Error (RMSE)": LSTM_ERRORS["rmse"],
        "Mean Absolute Error (MAE)": LSTM_ERRORS["mae"],
    }


# Endpoint para exibir a acurácia pré-calculada do modelo RSI
@app.get("/rsi-accuracy/")
async def get_rsi_accuracy():
    return {"Acurácia do Modelo (Neural Network - RSI)": RSI_ACCURACY["accuracy"]}
