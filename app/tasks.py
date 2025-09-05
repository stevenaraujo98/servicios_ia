from celery import Celery
from .consts import BROKER_URL, RESULT_BACKEND
from .projects.objetivo import calificate_objectives_gen_esp

# Creamos la instancia de la aplicación Celery
celery_app = Celery('tasks', broker=BROKER_URL, backend=RESULT_BACKEND)

# --- Definición de la Tarea ---
# El decorador @celery_app.task convierte esta función en una tarea de fondo.
@celery_app.task
def run_objective_evaluation_task(model_name: str, objetivo: str, objetivos_especificos: list):
    """
    Esta es la tarea que se ejecutará en segundo plano en un worker de Celery.
    Simplemente llama a tu función original con los argumentos recibidos.
    """
    print(f"Worker de Celery iniciando evaluación para el modelo: {model_name}")
    
    # Aquí se ejecuta tu función pesada. Puede tardar varios minutos.
    alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(
        model_name, objetivo, objetivos_especificos
    )

    print(f"Approved: {alineacion_aprobada}")
    print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
    print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
    print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")

    # El valor que retornas aquí es el que se guardará como resultado de la tarea.
    response_data = {
        "joint_evaluation": evaluacion_conjunta,
        "individual_evaluation": evaluacion_individual
    }
    
    print(f"Worker de Celery finalizó la evaluación para el modelo: {model_name}")
    return response_data
