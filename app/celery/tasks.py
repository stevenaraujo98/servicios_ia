import redis
import json
from ..projects.objetivos_gen_spe import calificate_objectives_gen_esp
from .worker import celery_app
from app.consts import REDIS_PORT

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
    Simplemente llama a tu función original con los argumentos recibidos.
    """
    task_id = self.request.id
    print(f"Worker: Iniciando evaluación para la tarea {task_id} con el modelo: {model_name}")

    
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

        

        # 2. Publish result to Redis
        print(f"[DIAGNÓSTICO] Worker: Attempting to publish result to Redis for task: {task_id}...")
        try:
            # BEST PRACTICE: Instantiate client inside the task
            redis_client = redis.Redis(host='redis', port=REDIS_PORT, db=0)
            
            # --- ¡PASO CLAVE! Publicar el resultado en el canal 'task_results' ---
            # El formato del mensaje es "task_id:resultado_en_json"
            message = f"{task_id}:{json.dumps(response_data)}"
            redis_client.publish("task_results", message)
            print(f"[DIAGNÓSTICO] Worker: Result published successfully to Redis.")
        except Exception as e:
            print(f"[DIAGNÓSTICO ERROR] Worker: FAILED to publish result to Redis. Error: {e}")
            raise  # Re-raise the exception to see it clearly
        
        print(f"Worker de Celery finalizó la evaluación para el modelo: {model_name}")
        print(f"Worker: Finalizó y publicó el resultado para la tarea: {task_id}")
        return response_data

    except Exception as e:
        print(f"Worker: Error en la tarea {task_id}: {e}")
        # Opcional: Publicar también los mensajes de error
        error_payload = json.dumps({"status": "FAILURE", "error": str(e)})
        message = f"{task_id}:{error_payload}"
        
        # Re-lanza la excepción para que Celery marque la tarea como fallida
        try:
            redis_client = redis.Redis(host='redis', port=REDIS_PORT, db=0)
            redis_client.publish('task_results', message)
            print(f"[DIAGNÓSTICO] Worker: FAILURE message published successfully to Redis.")
        except Exception as pub_e:
            print(f"[DIAGNÓSTICO ERROR] Worker: FAILED to publish FAILURE message to Redis. Error: {pub_e}")
        
        raise e

