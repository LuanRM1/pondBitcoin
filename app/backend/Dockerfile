# Usar uma imagem base Python slim
FROM python:3.9-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar e instalar as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

# Expor a porta que o FastAPI usará
EXPOSE 8000

# Comando para iniciar o aplicativo com python3
CMD ["python3", "main.py"]
