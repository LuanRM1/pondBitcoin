import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


# Carregar o modelo neural treinado
@st.cache_resource
def load_trained_model():
    model = load_model(
        "neural_network_model.h5"
    )  # Atualize o caminho conforme necessário
    return model


nn_model = load_trained_model()


# Função para calcular o RSI
def calculate_rsi(data, window=14):
    delta = data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Função para carregar dados e calcular indicadores
@st.cache_data
def get_data():
    avax = yf.Ticker("AVAX-USD")
    avax_data = avax.history(period="5y")

    # Calcular RSI
    avax_data["RSI"] = calculate_rsi(avax_data)

    # Calcular médias móveis
    avax_data["SMA_30"] = avax_data["Close"].rolling(window=30).mean()
    avax_data["SMA_100"] = avax_data["Close"].rolling(window=100).mean()

    # Direção do preço (0 ou 1) - se o preço fechar acima ou abaixo do valor anterior
    avax_data["Price_Direction"] = np.where(
        avax_data["Close"].shift(-1) > avax_data["Close"], 1, 0
    )

    avax_data.dropna(
        inplace=True
    )  # Remover valores nulos após o cálculo dos indicadores

    # Características para o modelo
    features = avax_data[["RSI", "SMA_30", "SMA_100"]]

    # Normalizar os dados
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return avax_data, features_scaled, scaler


# Função para prever com o modelo neural
def predict_with_neural_network(model, data, days=30):
    X_test = np.array(data[-days:])  # Pegar os últimos dados para prever
    X_test = X_test.reshape(
        (X_test.shape[0], X_test.shape[1], 1)
    )  # Redimensionar para o formato esperado pelo modelo

    predictions = model.predict(X_test)
    return predictions.flatten()


# Função para recomendar compra ou venda baseado no RSI
def rsi_recommendation(rsi_value):
    if rsi_value > 70:
        return "Overbought: Consider Selling"
    elif rsi_value < 30:
        return "Oversold: Consider Buying"
    else:
        return "Hold: No Strong Signal"


# Função para gerar o gráfico com sinais de compra e venda
def plot_rsi_chart(data, future_predictions, scaler):
    fig = go.Figure()

    # Adicionar o gráfico de preço
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price")
    )

    # Identificar pontos de compra e venda
    buy_signals = data[data["RSI"] < 30]["Close"]
    sell_signals = data[data["RSI"] > 70]["Close"]

    # Plotar os pontos de compra
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals,
            mode="markers",
            name="Buy Signal",
            marker=dict(color="green", size=10, symbol="triangle-up"),
        )
    )

    # Plotar os pontos de venda
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals,
            mode="markers",
            name="Sell Signal",
            marker=dict(color="red", size=10, symbol="triangle-down"),
        )
    )

    # Adicionar as previsões futuras
    future_dates = [
        data.index[-1] + pd.Timedelta(days=i)
        for i in range(1, len(future_predictions) + 1)
    ]

    # Preencher zeros nas colunas adicionais para inverter a escala
    future_predictions_reshaped = np.zeros((future_predictions.shape[0], 3))
    future_predictions_reshaped[:, 0] = (
        future_predictions  # Preencher apenas a coluna de preços
    )

    # Inverte a escala
    future_predictions = scaler.inverse_transform(future_predictions_reshaped)[
        :, 0
    ]  # Obtem apenas a coluna de preços

    # Plotar as previsões futuras
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=future_predictions, mode="lines", name="Predicted Prices"
        )
    )

    # Atualizar layout
    fig.update_layout(
        title="AVAX-USD Price with RSI Buy/Sell Signals and Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        height=600,
    )

    return fig


# Interface principal do Streamlit
def main():
    st.title("AVAX-USD RSI Analysis, Neural Network Predictions and Recommendations")

    # Carregar os dados e calcular indicadores
    avax_data, features_scaled, scaler = get_data()

    # Obter o RSI mais recente
    latest_rsi = avax_data["RSI"].iloc[-1]
    st.write(f"Latest RSI Value: {latest_rsi:.2f}")

    # Exibir recomendação com base no RSI
    recommendation = rsi_recommendation(latest_rsi)
    st.subheader(f"Recommendation: {recommendation}")

    # Parâmetro de quantos dias prever
    days_to_predict = st.slider("Number of days to predict", 1, 60, 30)

    # Prever com o modelo neural
    future_predictions = predict_with_neural_network(
        nn_model, features_scaled, days=days_to_predict
    )

    # Plotar gráfico com sinais de compra e venda e previsões
    fig = plot_rsi_chart(avax_data, future_predictions, scaler)
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
