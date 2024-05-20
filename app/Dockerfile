# Use uma imagem base Python
FROM python:3.8-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt requirements.txt

# Instala as dependências Python especificadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo do diretório atual para o diretório de trabalho no contêiner
COPY . .

# Expor a porta em que o Flask está sendo executado
EXPOSE 5000

# Comando para executar o aplicativo quando o contêiner for iniciado
CMD ["python", "main.py"]
