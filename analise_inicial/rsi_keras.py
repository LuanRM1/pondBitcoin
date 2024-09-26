import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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
avax_data = avax.history(period="5y")

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

# Normalizar os dados
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 4. Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 5. Construção do Modelo de Rede Neural
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compilar o modelo
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento do modelo
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# 6. Avaliação do Modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia do Modelo (Neural Network): {accuracy:.2f}")

# 7. Previsão do Momento Atual
current_data = avax_data[["RSI", "SMA_30", "SMA_100"]].iloc[-1].values.reshape(1, -1)
current_data_scaled = scaler.transform(current_data)
current_prediction = model.predict(current_data_scaled)
print("Momento Atual:", "Compra" if current_prediction > 0.5 else "Venda")
