# Implementando as melhorias sugeridas no modelo

import yfinance as yf
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


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


# Coleta de dados históricos
avax = yf.Ticker("AVAX-USD")
avax_data = avax.history(period="5y")

# Criação de Indicadores Técnicos
avax_data["RSI"] = calculate_rsi(avax_data)
avax_data["SMA_30"] = avax_data["Close"].rolling(window=30).mean()
avax_data["SMA_100"] = avax_data["Close"].rolling(window=100).mean()
avax_data["Momentum"] = avax_data["Close"] - avax_data["Close"].shift(4)
avax_data["Price_Direction"] = np.where(
    avax_data["Close"].shift(-1) > avax_data["Close"], 1, 0
)

# Preparação do Dataset
avax_data.dropna(inplace=True)
features = avax_data[["RSI", "SMA_30", "SMA_100", "Momentum"]]
target = avax_data["Price_Direction"]

# Verificar o balanceamento das classes
print("Distribuição Original das Classes:", Counter(target))

# Aplicar SMOTE para balancear as classes
smote = SMOTE(random_state=42)
features_resampled, target_resampled = smote.fit_resample(features, target)

# Verificar o balanceamento após SMOTE
print("Distribuição das Classes após SMOTE:", Counter(target_resampled))

# Filtrar as linhas onde não há valores nulos nas features (antes de normalizar)
non_null_indices = features_resampled.index

# Garantir que o target também esteja alinhado
features_resampled = features_resampled.loc[non_null_indices]
target_resampled = target_resampled.loc[non_null_indices]

# Normalizar os dados
scaler = StandardScaler()
features_resampled = scaler.fit_transform(features_resampled)

# Fazer o train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features_resampled, target_resampled, test_size=0.2, random_state=42
)

# Construção do Modelo Neural com Dropout e menos neurônios
model = Sequential()
model.add(
    Dense(64, input_dim=X_train.shape[1], activation="relu")
)  # Reduzir de 128 para 64
model.add(Dropout(0.5))  # Aumentar o Dropout para 0.5
model.add(Dense(32, activation="relu"))  # Reduzir de 64 para 32
model.add(Dense(16, activation="relu"))  # Nova camada de 16 neurônios
model.add(Dense(1, activation="sigmoid"))

# Compilar o modelo
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento do modelo (reduzir épocas e aumentar o batch_size)
model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1)

# Avaliação do Modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia do Modelo Neural Melhorado: {accuracy:.2f}")

# Cross-validation com Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Avaliar Random Forest
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Acurácia do Random Forest: {rf_accuracy:.2f}")

# Cross-validation no Random Forest
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"Acurácia Média no Cross-Validation (Random Forest): {np.mean(cv_scores):.2f}")

# Verificar a distribuição das classes no conjunto de teste
print("Distribuição no Conjunto de Teste:", Counter(y_test))
