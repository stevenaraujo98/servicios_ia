from celery import Celery
# from ..consts import broker_url, result_backend

# Creamos la instancia de la aplicación Celery
celery_app = Celery('tasks', include=['app.celery.tasks'])
# celery_app = Celery('tasks', broker=broker_url, backend=result_backend)

# Le decimos a Celery que cargue toda su configuración desde nuestro archivo consts.
# Esto incluye el broker_url, result_backend y el nuevo result_expires.
celery_app.config_from_object('app.consts')