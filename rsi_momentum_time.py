import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Função para calcular RSI
def calculate_rsi(data, window=14):
    delta = data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# 1. Coleta de Dados Históricos (5 anos)
avax = yf.Ticker("AVAX-USD")
avax_data = avax.history(period="5y")  # Alterado para 5 anos

# 2. Criação de Indicadores Técnicos
avax_data["RSI"] = calculate_rsi(avax_data)
avax_data["SMA_30"] = avax_data["Close"].rolling(window=30).mean()
avax_data["SMA_100"] = avax_data["Close"].rolling(window=100).mean()
avax_data["Price_Direction"] = np.where(
    avax_data["Close"].shift(-1) > avax_data["Close"], 1, 0
)

# 3. Preparação do Dataset
avax_data.dropna(inplace=True)
features = avax_data[["RSI", "SMA_30", "SMA_100"]]
target = avax_data["Price_Direction"]

# 4. Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 5. Treinamento de um Modelo de Machine Learning
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Avaliação do Modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do Modelo: {accuracy:.2f}")

# 7. Previsão do Momento Atual
# 7. Previsão do Momento Atual
current_data = avax_data[["RSI", "SMA_30", "SMA_100"]].iloc[-1].values.reshape(1, -1)
current_data_df = pd.DataFrame(current_data, columns=["RSI", "SMA_30", "SMA_100"])
current_prediction = model.predict(current_data_df)
print("Momento Atual:", "Compra" if current_prediction == 1 else "Venda")
