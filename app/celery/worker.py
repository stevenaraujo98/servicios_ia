from celery import Celery
# from ..consts import BROKER_URL, RESULT_BACKEND

# Creamos la instancia de la aplicación Celery
celery_app = Celery('tasks', include=['app.celery.tasks'])
# celery_app = Celery('tasks', broker=BROKER_URL, backend=RESULT_BACKEND)

# Le decimos a Celery que cargue toda su configuración desde nuestro archivo consts.
# Esto incluye el BROKER_URL, RESULT_BACKEND y el nuevo CELERY_RESULT_EXPIRES.
celery_app.config_from_object('app.consts')