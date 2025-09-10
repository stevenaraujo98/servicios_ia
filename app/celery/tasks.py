import redis
import json
from .worker import celery_app
from app.projects.objetivos_gen_spec.logic import calificate_objectives_gen_esp
from app.consts import REDIS_HOST, REDIS_PORT, REDIS_STORE_DB_INDEX

# --- AÑADIDO: Conexión a Redis para publicar resultados ---
# Usamos el nombre del servicio de Docker 'redis' como el host.
# decode_responses=True es importante para que los mensajes sean strings.
# redis_client = redis.Redis(host='redis', port=REDIS_PORT, db=0, decode_responses=True)

# --- Definición de la Tarea ---
# El decorador @celery_app.task convierte esta función en una tarea de fondo.
# 'bind=True' es crucial para poder acceder al 'self' de la tarea, que contiene el ID.
@celery_app.task(bind=True)
def run_objective_evaluation_task(self, model_name: str, objetivo: str, objetivos_especificos: list):
    """
    Esta es la tarea que se ejecutará en segundo plano en un worker de Celery.
    """
    task_id = self.request.id
    print(f"Worker: Iniciando evaluación para la tarea {task_id} con el modelo: {model_name}")
    
    # BEST PRACTICE: Instanciamos el cliente una sola vez al inicio de la tarea
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_STORE_DB_INDEX)

    try:
        # Aquí se ejecuta tu función pesada. Puede tardar varios minutos.
        alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(
            model_name, objetivo, objetivos_especificos
        )

        # El valor que retornas aquí es el que se guardará como resultado de la tarea.
        response_data = {
            "joint_evaluation": evaluacion_conjunta,
            "individual_evaluation": evaluacion_individual
        }
        
        print(f"Approved: {alineacion_aprobada}")
        print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
        print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
        print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")

        # Preparamos un payload JSON unificado para éxito.
        payload = json.dumps({
            "task_id": task_id,
            "status": "SUCCESS",
            "result": response_data
        })

        # Publicamos el resultado en el canal 'task_results'
        print(f"[DIAGNÓSTICO] Worker: Publicando resultado para la tarea: {task_id}...")
        redis_client.publish("task_results", payload)
        print(f"[DIAGNÓSTICO] Worker: Resultado publicado exitosamente en Redis.")
        
        # El return sigue siendo importante para que Celery guarde el resultado en el backend
        return response_data

    except Exception as e:
        print(f"Worker: Error en la tarea {task_id}: {e}")
        
        # --- ¡CAMBIO CLAVE! ---
        # Preparamos un payload JSON unificado para fallos.
        error_payload = json.dumps({
            "task_id": task_id,
            "status": "FAILURE",
            "error": str(e)
        })

        # Publicamos el mensaje de error
        try:
            redis_client.publish('task_results', error_payload)
            print(f"[DIAGNÓSTICO] Worker: Mensaje de FALLO publicado exitosamente en Redis.")
        except Exception as pub_e:
            print(f"[DIAGNÓSTICO ERROR] Worker: FALLÓ al publicar mensaje de error en Redis. Error: {pub_e}")
        
        # Re-lanzamos la excepción para que Celery marque la tarea como fallida
        raise e
