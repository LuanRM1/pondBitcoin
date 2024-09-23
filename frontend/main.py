import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# URL do backend FastAPI
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Previsão AVAX-USD", layout="wide")

st.title("Previsão AVAX-USD")

# Barra lateral para entrada do usuário
st.sidebar.header("Configurações de Previsão")
days_to_predict = st.sidebar.slider("Número de dias para prever", 1, 60, 30)


# Funções para obter previsões (inalterado)
def get_lstm_predictions(days):
    response = requests.get(f"{BACKEND_URL}/predict/lstm?days={days}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Falha ao buscar previsões LSTM")
        return None


def get_rsi_predictions(days):
    response = requests.get(f"{BACKEND_URL}/predict/rsi?days={days}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Falha ao buscar previsões RSI")
        return None


# Função para obter logs de uso do modelo
def get_usage_logs():
    response = requests.get(f"{BACKEND_URL}/model_usage_logs")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Falha ao buscar logs de uso")
        return None


# Função para plotar previsões
def plot_predictions(future_dates, future_predictions, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name="Previsões"))
    fig.update_layout(title=title, xaxis_title="Data", yaxis_title="Preço")
    return fig


# Função para exibir métricas com explicações
def display_metrics(metrics, model_name):
    st.subheader(f"Métricas de Desempenho do {model_name}")

    if model_name == "LSTM":
        st.write("Interpretando as Métricas do LSTM:")
        st.write(
            "- Mean Squared Error (MSE): Diferença quadrada média entre os valores previstos e reais. Quanto menor, melhor."
        )
        st.write(
            "- Root Mean Squared Error (RMSE): Raiz quadrada do MSE, na mesma unidade da variável alvo. Quanto menor, melhor."
        )
        st.write(
            "- Mean Absolute Error (MAE): Diferença média absoluta entre os valores previstos e reais. Quanto menor, melhor."
        )

    elif model_name == "Rede Neural RSI":
        st.write("Interpretando as Métricas da Rede Neural RSI:")
        st.write("- Precisão: Proporção de previsões corretas. Quanto maior, melhor.")
        st.write("- Perda: Medida do erro de previsão. Quanto menor, melhor.")

    for key, value in metrics.items():
        st.write(f"{key}: {value}")


# Função para verificar a recomendação do RSI
def verificar_recomendacao(rsi_value):
    if rsi_value > 70:
        return "Venda"
    elif rsi_value < 30:
        return "Compra"
    else:
        return "Hold"


# Exibir recomendação com destaque
def exibir_recomendacao(rsi_value):
    recomendacao = verificar_recomendacao(rsi_value)
    if recomendacao == "Venda":
        st.error(
            f"Recomendação: {recomendacao} - RSI ({rsi_value:.2f}) está acima de 70, indicando sobrecompra."
        )
    elif recomendacao == "Compra":
        st.success(
            f"Recomendação: {recomendacao} - RSI ({rsi_value:.2f}) está abaixo de 30, indicando sobrevenda."
        )
    else:
        st.warning(
            f"Recomendação: {recomendacao} - RSI ({rsi_value:.2f}) está entre 30 e 70, sugerindo manter (Hold)."
        )


if st.button("Gerar Previsões"):
    col1, col2 = st.columns(2)

    with col1:
        lstm_data = get_lstm_predictions(days_to_predict)
        if lstm_data:
            st.subheader("Previsões LSTM")
            fig_lstm = plot_predictions(
                lstm_data["dates"],
                lstm_data["predictions"],
                "Previsão de Preço AVAX-USD (Baseado em LSTM)",
            )
            st.plotly_chart(fig_lstm)
            display_metrics(lstm_data["metrics"], "LSTM")

            st.write("Explicação do Modelo LSTM:")
            st.write(
                "O modelo LSTM (Long Short-Term Memory) é um tipo de rede neural recorrente que pode capturar dependências de longo prazo em dados de séries temporais. Ele é especialmente útil para prever valores futuros com base em padrões históricos."
            )
            st.write("Como usar as previsões do LSTM:")
            st.write("1. Observe a tendência dos preços previstos.")
            st.write("2. Compare a tendência prevista com a tendência histórica.")
            st.write("3. Use o RMSE como referência para o erro médio nas previsões.")
            st.write(
                "4. Lembre-se de que previsões para prazos mais longos tendem a ser menos precisas."
            )

    with col2:
        rsi_data = get_rsi_predictions(days_to_predict)
        if rsi_data:
            st.subheader("Previsões Baseadas em Rede Neural RSI")
            fig_rsi = plot_predictions(
                rsi_data["dates"],
                rsi_data["predictions"],
                "Previsão de Preço AVAX-USD (Baseado em Rede Neural RSI)",
            )
            st.plotly_chart(fig_rsi)
            exibir_recomendacao(rsi_data["latest_rsi"])
            display_metrics(rsi_data["metrics"], "Rede Neural RSI")

            st.subheader("Informações do RSI")
            st.write(f"Último Valor RSI: {rsi_data['latest_rsi']:.2f}")

            # Exibir recomendação com destaque

            st.write("Explicação do Modelo Neural RSI:")
            st.write(
                "Este modelo combina o Índice de Força Relativa (RSI) com uma rede neural para prever preços futuros. O RSI é um indicador de momento que mede a magnitude das mudanças recentes de preço para avaliar condições de sobrecompra ou sobrevenda."
            )
            st.write("Como interpretar as previsões do RSI:")
            st.write("1. Os valores do RSI variam de 0 a 100.")
            st.write(
                "2. RSI > 70 normalmente indica condições de sobrecompra (possível sinal de venda)."
            )
            st.write(
                "3. RSI < 30 normalmente indica condições de sobrevenda (possível sinal de compra)."
            )
            st.write(
                "4. A rede neural usa o RSI e outras características para fazer previsões de preço."
            )
            st.write(
                "5. Considere tanto o valor do RSI quanto a tendência de preço prevista na sua análise."
            )

# Exibir logs de uso do modelo
st.header("Logs de Uso do Modelo")
usage_logs = get_usage_logs()
if usage_logs:
    df_logs = pd.DataFrame(usage_logs)
    st.write("Logs de uso do modelo:")
    st.dataframe(df_logs)
else:
    st.write("Nenhum log de uso encontrado.")
