import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 1. Coleta de Dados Históricos (5 anos)
avax = yf.Ticker("AVAX-USD")
avax_data = avax.history(period="5y")

# Usar apenas a coluna 'Close' para previsão
data = avax_data[["Close"]]

# 2. Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Criar as sequências para o modelo LSTM
sequence_length = 60  # Usaremos 60 dias de histórico para prever o próximo dia
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length : i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Redimensionar X para ter a forma [amostras, sequência temporal, número de features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Divisão dos Dados (80% treino, 20% teste)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Construção do Modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# Treinamento do modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# 6. Fazer Previsões
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

# Previsão dos próximos 30 dias
last_sequence = X_test[-1]  # Última sequência do conjunto de teste
predictions = []

for _ in range(30):
    pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
    predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred)[-sequence_length:]
    last_sequence = last_sequence.reshape(sequence_length, 1)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Preços reais de teste para comparação
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. Cálculo das Métricas do Modelo
mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_prices, predicted_prices)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 8. Plotar o Gráfico de Previsão vs Real, incluindo os próximos 30 dias
plt.figure(figsize=(14, 7))
plt.plot(real_prices, label="Preço Real")
plt.plot(predicted_prices, label="Preço Previsto", linestyle="dashed")

# Adicionar as previsões dos próximos 30 dias ao gráfico
future_days = np.arange(len(real_prices), len(real_prices) + 30)
plt.plot(
    future_days,
    predictions,
    label="Previsão 30 Dias Futuros",
    linestyle="dotted",
    color="green",
)

plt.title("Previsão de Preços de Avalanche (AVAX) com LSTM")
plt.xlabel("Dias")
plt.ylabel("Preço (USD)")
plt.legend()
plt.show()
