from fastapi import APIRouter, HTTPException
from typing import Union

from .logic import calificate_objectives_gen_esp_simple

# --- Importaciones de Celery tasks ---
from app.celery.tasks import run_objective_evaluation_task

# --- Importaciones de tu proyecto ---
from app.consts import stages
from app.validations import validate_min_length, validate_not_empty, clean_text, validation_response_redis
from app.entities import ItemModelContentObjectives, TaskCreationResponse, FullEvaluationResponse, ItemContentObjectives

# --- Importaciones de Celery tasks ---
from app.celery.tasks import celery_app
from celery.result import AsyncResult

objetivo_gen_spe_router = APIRouter()

# # Calificador Objetivos
# @objetivo_gen_spe_router.get("/")
# def read_objetivos():
#     objetivo = "Desarrollando un diseño que permita la visualización de la curva I-V de un panel PV mediante la implementación de un método práctico, programable, para que pueda ser replicado por estudiantes de pregrado."

#     objetivos_especificos = [
#         "Desarrollar un procedimiento para la determinación de la curva de operación I-V para la obtención de mediciones de forma automática.",
#         "Diseñar un prototipo escalable basado en el método seleccionado para el trazador de curvas I-V",
#         "Realizar las mediciones de corriente y voltaje de un panel PV que nos permitan la adquisición diferentes puntos de la curva."
#     ]

#     model_name = "gemma3"
#     alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(model_name, objetivo, objetivos_especificos)

#     print(f"Approved: {alineacion_aprobada}")
#     print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
#     print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
#     print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")

#     response_data = {
#         "joint_evaluation": evaluacion_conjunta,
#         "individual_evaluation": evaluacion_individual
#     }    
#     return response_data


@objetivo_gen_spe_router.post("/", response_model=FullEvaluationResponse)
def predict_objetivos(item: ItemModelContentObjectives, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    objetivos_especificos = item.specific_objectives
    if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
        raise ValueError("La lista de objetivos específicos no puede estar vacía, tampoco menos de 3 ni más de 4.")

    alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp_simple(model_name, objetivo, objetivos_especificos)

    print(f"Approved: {alineacion_aprobada}")
    print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
    print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
    print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")
    
    response_data = {
        "joint_evaluation": evaluacion_conjunta,
        "individual_evaluation": evaluacion_individual
    }
    
    return response_data

@objetivo_gen_spe_router.post("/model_name/{model_name}", response_model=FullEvaluationResponse)
def predict_objetivos(model_name: str, item: ItemContentObjectives, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    objetivos_especificos = item.specific_objectives
    if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
        raise ValueError("La lista de objetivos específicos no puede estar vacía, tampoco menos de 3 ni más de 4.")

    alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp_simple(model_name, objetivo, objetivos_especificos)

    print(f"Approved: {alineacion_aprobada}")
    print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
    print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
    print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")
    
    response_data = {
        "joint_evaluation": evaluacion_conjunta,
        "individual_evaluation": evaluacion_individual
    }
    
    return response_data

# Este endpoint ahora INICIA la tarea y responde inmediatamente.
@objetivo_gen_spe_router.post("/async", response_model=TaskCreationResponse, status_code=202)
def predict_objetivos_async(item: ItemModelContentObjectives):
    # 1. Validaciones (se mantienen igual)
    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    validate_min_length(objetivo, min_length=10)

    objetivos_especificos = item.specific_objectives
    if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
        raise HTTPException(status_code=400, detail="La lista de objetivos específicos debe contener entre 3 y 4 elementos.")

    # 2. Iniciar la tarea en segundo plano
    # En lugar de llamar a la función directamente, usamos .delay()
    # Esto envía la tarea a la cola de RabbitMQ y no espera.
    task = run_objective_evaluation_task.delay(
        model_name, objetivo, objetivos_especificos
    )

    # 3. Responder inmediatamente con el ID de la tarea Processing
    return {"task_id": task.id, "status": stages[0]}

# --- Obtener resultado de la tarea ---
@objetivo_gen_spe_router.get("/result/{task_id}", response_model=FullEvaluationResponse)
def get_task_result(task_id: str):
    """Obtiene el resultado de una tarea completada."""
    task_result = AsyncResult(task_id, app=celery_app)

    return validation_response_redis(task_result, FullEvaluationResponse)
