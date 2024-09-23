import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go


# Load and compile models
@st.cache_resource
def load_and_compile_models():
    # LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss="mean_squared_error")

    # Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    return lstm_model, rf_model


lstm_model, rf_model = load_and_compile_models()


def calculate_rsi(data, window=14):
    delta = data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@st.cache_data
def get_data():
    avax = yf.Ticker("AVAX-USD")
    avax_data = avax.history(period="5y")

    # Create technical indicators
    avax_data["RSI"] = calculate_rsi(avax_data)
    avax_data["SMA_30"] = avax_data["Close"].rolling(window=30).mean()
    avax_data["SMA_100"] = avax_data["Close"].rolling(window=100).mean()
    avax_data["Price_Direction"] = np.where(
        avax_data["Close"].shift(-1) > avax_data["Close"], 1, 0
    )

    avax_data.dropna(inplace=True)

    # Prepare the dataset
    features = avax_data[["RSI", "SMA_30", "SMA_100"]]
    target = avax_data["Price_Direction"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Scale the data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(avax_data[["Close"]])

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Define split for LSTM data
    split = int(len(X) * 0.8)

    X_train_lstm, X_test_lstm = X[:split], X[split:]
    y_train_lstm, y_test_lstm = y[:split], y[split:]

    return (
        avax_data,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_lstm,
        X_test_lstm,
        y_train_lstm,
        y_test_lstm,
        scaler,
    )


def train_lstm_model(X_train, y_train, X_test, y_test):
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Evaluate the LSTM model
    mse = mean_squared_error(y_test, lstm_model.predict(X_test))
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, lstm_model.predict(X_test))

    return mse, rmse, mae


def train_rf_model(X_train, y_train):
    rf_model.fit(X_train, y_train)

    # Evaluate the Random Forest model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


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


def predict_rf_future(X_test, days=1):
    current_data = X_test.iloc[-1].values.reshape(1, -1)
    current_prediction = rf_model.predict(current_data)
    return current_prediction[0]


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


def main():
    st.title("AVAX-USD Price Prediction")

    tab1, tab2, tab3 = st.tabs(
        ["LSTM Predictions", "RSI Predictions", "Model Performance"]
    )

    with tab1:
        st.header("LSTM-based Predictions")
        days_lstm = st.slider("Number of days to predict (LSTM)", 1, 60, 30)

        (
            avax_data,
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_lstm,
            X_test_lstm,
            y_train_lstm,
            y_test_lstm,
            scaler,
        ) = get_data()

        # Train LSTM model
        mse, rmse, mae = train_lstm_model(
            X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm
        )

        # Make LSTM predictions
        future_predictions_lstm = predict_lstm_future(scaler, X_test_lstm, days_lstm)
        future_dates_lstm = [
            (avax_data.index[-1] + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, days_lstm + 1)
        ]

        fig_lstm = plot_predictions(
            avax_data.index.strftime("%Y-%m-%d").tolist(),
            avax_data["Close"].tolist(),
            future_dates_lstm,
            future_predictions_lstm.flatten(),
            "AVAX-USD Price Prediction (LSTM-based)",
        )
        st.plotly_chart(fig_lstm)

    with tab2:
        st.header("RSI-based Predictions")
        days_rsi = st.slider("Number of days to predict (RSI)", 1, 60, 30)

        (
            avax_data,
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_lstm,
            X_test_lstm,
            y_train_lstm,
            y_test_lstm,
            scaler,
        ) = get_data()

        # Train Random Forest model
        accuracy = train_rf_model(X_train, y_train)

        # Make RSI-based predictions
        current_prediction = predict_rf_future(X_test, days_rsi)
        future_dates_rsi = [
            (avax_data.index[-1] + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, days_rsi + 1)
        ]
        future_predictions_rsi = [
            1 if i == 1 else 0 for i in [current_prediction] * days_rsi
        ]

        fig_rsi = plot_predictions(
            avax_data.index.strftime("%Y-%m-%d").tolist(),
            avax_data["Close"].tolist(),
            future_dates_rsi,
            avax_data["Close"].iloc[-1]
            * (1 + np.cumsum(future_predictions_rsi) * 0.01),
            "AVAX-USD Price Prediction (RSI-based)",
        )
        st.plotly_chart(fig_rsi)

    with tab3:
        st.header("Model Performance")
        st.subheader("LSTM Model Errors")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

        st.subheader("Random Forest Model Accuracy")
        st.write(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
