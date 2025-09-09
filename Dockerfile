FROM python:3.12.11-slim-bullseye

# Establece el directorio de trabajo "cd /code"
WORKDIR /code

# Instala dependencias del sistema.
# build-essential es necesario para paquetes de Python que compilan código C/C++.
# Combinamos los comandos para reducir el número de capas en la imagen.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl redis-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copia primero el archivo de requerimientos e instala las dependencias.
# Esto aprovecha el caché de Docker: si no cambias requirements.txt, esta capa no se reconstruirá.
# COPY ./requirements.txt .
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt
 
# Descarga el modelo de spacy, también se beneficia del caché.
RUN python -m spacy download en_core_web_lg
RUN python -m nltk.downloader stopwords
 
# Copia el resto del código de la aplicación.
# COPY ./app ./app
COPY ./app /code/app

# Comando para ejecutar la aplicación en producción (sin --reload)
# Uvicorn es el servidor ASGI recomendado para FastAPI.
# , "--workers", "2" https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker # no se usa por el recurso
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
