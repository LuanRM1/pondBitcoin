import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


# Carregar modelo treinado
@st.cache_resource
def load_trained_model():
    model = load_model("lstm_model.h5")  # Caminho para o modelo
    return model


lstm_model = load_trained_model()


# Função para calcular previsões futuras
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


# Função para plotar previsões
def plot_predictions(
    historical_dates, historical_prices, future_dates, future_predictions, title
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=historical_dates, y=historical_prices, name="Historical")
    )
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name="Predictions"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


# Carregar dados da AVAX-USD
@st.cache_data
def get_data():
    avax = yf.Ticker("AVAX-USD")
    avax_data = avax.history(period="5y")

    # Escalar os dados para o LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(avax_data[["Close"]])

    sequence_length = 60
    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return avax_data, X, scaler


# Interface principal
def main():
    st.title("AVAX-USD Price Prediction with LSTM")

    # Parâmetro de quantos dias prever
    days_lstm = st.slider("Number of days to predict (LSTM)", 1, 60, 30)

    # Carregar dados e realizar previsões
    avax_data, X_test, scaler = get_data()
    future_predictions_lstm = predict_lstm_future(scaler, X_test, days_lstm)
    future_dates_lstm = [
        (avax_data.index[-1] + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, days_lstm + 1)
    ]

    # Plotar os resultados
    fig_lstm = plot_predictions(
        avax_data.index.strftime("%Y-%m-%d").tolist(),
        avax_data["Close"].tolist(),
        future_dates_lstm,
        future_predictions_lstm.flatten(),
        "AVAX-USD Price Prediction (LSTM-based)",
    )
    st.plotly_chart(fig_lstm)


if __name__ == "__main__":
    main()
