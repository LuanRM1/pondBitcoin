# Usar uma imagem base Python slim
FROM python:3.9-slim

# Instalar dependências adicionais
RUN apt-get update && apt-get install -y build-essential

# Definir o diretório de trabalho
WORKDIR /app

# Copiar e instalar as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

# Expor a porta que o Streamlit usará
EXPOSE 8501

# Comando para iniciar o aplicativo
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
