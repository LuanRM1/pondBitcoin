import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


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


# 1. Baixar dados históricos do Avalanche (AVAX)
avax = yf.Ticker("AVAX-USD")
avax_data = avax.history(period="1y")  # Dados de 1 ano

# 2. Calcular o RSI
avax_data["RSI"] = calculate_rsi(avax_data)

# 3. Definir sinais de compra e venda
avax_data["Buy Signal"] = avax_data["RSI"] < 30
avax_data["Sell Signal"] = avax_data["RSI"] > 70

# 4. Visualizar os dados e os sinais
plt.figure(figsize=(14, 7))

# Preço de Fechamento
plt.plot(avax_data["Close"], label="AVAX Price", color="blue")

# Sinais de Compra
plt.plot(
    avax_data.index,
    avax_data["Close"],
    "^",
    markersize=10,
    color="green",
    lw=0,
    label="Buy Signal",
    markevery=avax_data["Buy Signal"],
)

# Sinais de Venda
plt.plot(
    avax_data.index,
    avax_data["Close"],
    "v",
    markersize=10,
    color="red",
    lw=0,
    label="Sell Signal",
    markevery=avax_data["Sell Signal"],
)

plt.title("AVAX Price with RSI Buy/Sell Signals")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Exibir as últimas entradas de sinais
print(avax_data[["Close", "RSI", "Buy Signal", "Sell Signal"]].tail(10))
