# AVAX-USD Prediction App

Este projeto é uma aplicação de previsão de preços de AVAX-USD utilizando **FastAPI** para o backend e **Streamlit** para o frontend. Ele prevê o preço da AVAX-USD usando modelos LSTM e uma rede neural baseada no RSI. A aplicação é completamente containerizada usando **Docker Compose** para facilitar a execução.

## Requisitos

Certifique-se de que os seguintes pré-requisitos estão instalados no seu sistema:

- **Docker**: [Instalar Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Instalar Docker Compose](https://docs.docker.com/compose/install/)

## Instruções de Instalação

Siga os passos abaixo para configurar e rodar a aplicação:

### 1. Clone o Repositório

Primeiro, clone o repositório para sua máquina local:

```bash
git clone https://github.com/LuanRM1/pondBitcoin.git
```

Navegue até o diretório do projeto:

```bash
cd pondBitcoin
```

### 2. Estrutura do Projeto

A estrutura do projeto é a seguinte:

```bash
app/
├── backend/
│   ├── cache.py
│   ├── controllers/
│   │   └── prediction_controller.py
│   ├── lstm_model.h5
│   ├── main.py
│   ├── model_metrics.json
│   ├── model_usage_logs.json
│   ├── neural_network_model.h5
│   ├── requirements.txt
│   ├── routers/
│   │   └── prediction_router.py
│   └── utils/
│       └── logger.py
├── frontend/
│   ├── main.py
│   └── requirements.txt
└── docker-compose.yml
```

### 3. Rodar a Aplicação com Docker Compose

Depois de clonar o repositório, você pode usar o Docker Compose para rodar a aplicação. O **Docker Compose** gerencia tanto o backend quanto o frontend. Para fazer isso, siga os passos:

#### 3.1. Construir e Subir os Contêineres

No diretório raiz do projeto (app) (onde está o arquivo `docker-compose.yml`), execute o seguinte comando:

```bash
docker-compose up --build
```

Este comando irá:

- **Build** as imagens Docker tanto para o backend (FastAPI) quanto para o frontend (Streamlit).
- **Up** os contêineres, expondo as portas necessárias para acessar a aplicação.

#### 3.2. Acessar o Frontend

Após rodar o Docker Compose, o **frontend** estará disponível no seu navegador em:

```
http://localhost:8501
```

### 4. Parar a Aplicação

Para parar os contêineres rodando, use o seguinte comando no terminal:

```bash
docker-compose down
```

Este comando irá parar e remover os contêineres que estão rodando.

### **1. Previsão com LSTM**

O LSTM é a base do modelo, projetado para prever preços futuros da criptomoeda. Ele faz isso ao capturar padrões temporais de longo prazo em dados históricos. Isso é fundamental em mercados voláteis, como o de criptomoedas, onde eventos passados podem influenciar diretamente os preços futuros. Os dados utilizados para o treinamento do LSTM foram o preço de fechamento.

### **2. RSI como Indicador de Contexto**

O RSI, por sua vez, é uma ferramenta que auxilia na análise de momentum de mercado. Ele ajuda a identificar se uma criptomoeda está em condição de **sobrecompra** ou **sobrevenda**. Esse indicador é essencial para entender o estado do mercado no momento da previsão, permitindo ao usuário tomar decisões mais informadas.

Por exemplo:

- Se o LSTM prevê uma queda de preço, e o **RSI indica sobrecompra** (>70), isso pode reforçar a expectativa de que o mercado está propenso a uma correção.
- Se o LSTM prevê um aumento de preço e o **RSI indica sobrevenda** (<30), isso sugere que o mercado pode estar prestes a se recuperar.

Aqui está a explicação de cada um desses termos:

### **. RSI (Relative Strength Index)**

O **RSI** é um indicador técnico de **momentum** usado para medir a velocidade e a magnitude dos movimentos de preço recentes de um ativo. Ele varia de 0 a 100 e é utilizado principalmente para identificar condições de **sobrecompra** (quando o valor do RSI está acima de 70) e **sobrevenda** (quando o valor está abaixo de 30). Esses níveis indicam que o ativo pode estar prestes a sofrer uma correção ou reversão de tendência:

- **RSI > 70**: O ativo pode estar sobrecomprado, sugerindo uma possível queda de preço.
- **RSI < 30**: O ativo pode estar sobrevendido, indicando uma possível recuperação de preço.

## Os dados utilizados para o treinamento do RSI foram:

### **. SMA_30 (Simple Moving Average de 30 dias)**

A **SMA_30** é a **Média Móvel Simples** calculada sobre um período de **30 dias**. Ela é obtida somando os preços de fechamento dos últimos 30 dias e dividindo o total por 30. A SMA_30 é frequentemente usada para suavizar as flutuações de curto prazo nos preços e fornecer uma visão mais clara da tendência geral do mercado a curto prazo.

- Se o preço atual estiver **acima da SMA_30**, isso pode indicar uma **tendência de alta**.
- Se o preço estiver **abaixo da SMA_30**, pode sugerir uma **tendência de baixa**.

### **. SMA_100 (Simple Moving Average de 100 dias)**

A **SMA_100** é uma Média Móvel Simples calculada sobre um período de **100 dias**. Funciona de forma semelhante à SMA_30, mas como considera um período mais longo, oferece uma perspectiva mais **suave** e **longa** da tendência do mercado.

- A **SMA_100** é usada para identificar a tendência de **longo prazo**.
- Se o preço estiver **acima da SMA_100**, geralmente indica uma **tendência de alta** mais sustentada.
- Se o preço estiver **abaixo da SMA_100**, pode ser um sinal de **tendência de baixa**.

### **3. Auxílio na Tomada de Decisão**

O RSI oferece uma análise de **momentum do mercado**, mostrando potenciais pontos de reversão ou condições extremas. Ao combinar isso com a previsão do LSTM, o usuário obtém uma visão mais completa:

- **LSTM**: fornece a previsão com base em padrões de preços históricos.
- **RSI**: oferece uma leitura sobre o estado atual do mercado, o que ajuda a validar a previsão e ajustar a estratégia de compra ou venda.

### **4. Complementaridade na Análise**

A combinação do LSTM e do RSI fortalece a análise dos movimentos de preço. O LSTM prevê a tendência futura, enquanto o RSI acrescenta um componente técnico que pode ajudar a confirmar se essa tendência faz sentido, dada a condição atual do mercado. Para uma analise mai aprofundada verifique os notebooks na pasta estudo.

Aqui está a explicação em formato markdown, incluindo a menção ao pipeline de retraining e um cronograma:

---

## Planejamento para Retreinamento do Modelo com Novos Dados

O planejamento do retrain do modelo envolve a criação de um pipeline automatizado para coleta, processamento e integração de novos dados de mercado, como preços de criptomoedas e indicadores técnicos (RSI, SMA_30, SMA_100). O pipeline de retraining será configurado para rodar em intervalos predefinidos, coletando novos dados, realizando o pré-processamento (como normalização e cálculo de indicadores) e integrando-os ao modelo existente. O modelo pode ser atualizado de forma incremental ou por retraining completo, dependendo do volume de dados e da estratégia definida. Ferramentas como **Airflow** ou **Luigi** podem ser usadas para automatizar e gerenciar a sequência de etapas.

Após o retraining, o modelo será avaliado por meio de métricas como **MAE** e **RMSE** para garantir que as previsões sejam mais precisas. Esse pipeline será complementado com um sistema de deploy contínuo, garantindo que os novos modelos sejam implementados automaticamente em produção, com possibilidade de **rollback** se o desempenho não for satisfatório. Abaixo está um cronograma proposto para o retrain do modelo:

---

## Cronograma Proposto para Retraining

| Etapa                      | Frequência | Responsável                |
| -------------------------- | ---------- | -------------------------- |
| Coleta de novos dados      | Diária     | Automático                 |
| Pré-processamento de dados | Semanal    | Automático                 |
| Retreinamento do modelo    | Mensal     | Automático/Equipe de Dados |
| Avaliação do modelo        | Mensal     | Equipe de Dados            |
| Deploy do novo modelo      | Mensal     | Automático                 |
| Monitoramento contínuo     | Contínuo   | Automático                 |

---
