from celery import Celery
from ..consts import BROKER_URL, RESULT_BACKEND

# Creamos la instancia de la aplicación Celery
celery_app = Celery('tasks', broker=BROKER_URL, backend=RESULT_BACKEND)