FROM python:3.12.11-slim-bullseye

WORKDIR /code

# Instala dependencias del sistema.
# build-essential es necesario para paquetes de Python que compilan código C/C++.
# Combinamos los comandos para reducir el número de capas en la imagen.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copia primero el archivo de requerimientos e instala las dependencias.
# Esto aprovecha el caché de Docker: si no cambias requirements.txt, esta capa no se reconstruirá.
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt
 
# Descarga el modelo de spacy, también se beneficia del caché.
RUN python -m spacy download en_core_web_lg
 
# Copia el resto del código de la aplicación.
COPY ./app /code/app

# Ejecuta la aplicación. --host 0.0.0.0 es crucial para que sea accesible desde fuera del contenedor.
CMD ["fastapi", "run", "app/main.py", "--port", "8080", "--host", "0.0.0.0"]