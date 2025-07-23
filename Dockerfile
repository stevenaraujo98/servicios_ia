FROM python:3.12.11-slim-bullseye

RUN apt-get update 
WORKDIR /code

# Instala Rust (y build-essential por si acaso)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl build-essential && \
    apt-get clean

RUN pip install --upgrade pip

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

RUN python -m spacy download en_core_web_lg

CMD ["fastapi", "run", "app/main.py", "--port", "8080"]