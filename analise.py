import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# 1. Baixar dados históricos do Avalanche (AVAX)
avax = yf.Ticker("AVAX-USD")
avax_data = avax.history(period="2y")  # Baixar dados dos últimos 2 anos

# 2. Calcular a média móvel simples (SMA)
avax_data["SMA_30"] = avax_data["Close"].rolling(window=30).mean()

# 3. Visualizar os dados
plt.figure(figsize=(14, 7))
plt.plot(avax_data["Close"], label="Avalanche Price", color="blue")
plt.plot(avax_data["SMA_30"], label="30-Day SMA", color="orange")
plt.title("Avalanche (AVAX) Price with 30-Day SMA")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# 4. Identificar Sinais de Compra/Venda
avax_data["Signal"] = 0
avax_data["Signal"][30:] = np.where(
    avax_data["Close"][30:] > avax_data["SMA_30"][30:], 1, 0
)  # Compra se preço acima da SMA
avax_data["Position"] = avax_data["Signal"].diff()

# 5. Exibir os dados com sinais de compra e venda
print(avax_data[["Close", "SMA_30", "Position"]].tail(10))
