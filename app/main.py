from typing import Union, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult

from .projects.ods import ods_router
from .projects.patente import patente_router
from .projects.carrera import carrera_router
from .projects.objetivo import calificate_objective, calificate_objectives_gen_esp

# --- Importaciones de tu proyecto ---
from .validations import validate_min_length, validate_not_empty, clean_text
from .modelsEntity import ItemContent, ItemContentObjectives, ItemModelContent, ItemModelContentObjectives, PredictionResponseClassificationObjective, FullEvaluationResponse, TaskCreationResponse, TaskStatusResponse

# --- Importaciones de Celery ---
# Importamos la instancia de Celery y la tarea específica
from .tasks import celery_app, run_objective_evaluation_task

app = FastAPI()

origins = [
    "*",
    # "https://integradora.espol.edu.ec",
]

# Configuración de CORS 
app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "IA"}

app.include_router(
    patente_router,
    prefix="/predict/patente",
    tags=["Patente"],
)

app.include_router(
    ods_router,
    prefix="/predict/ods",
    tags=["ODS"],
)

app.include_router(
    carrera_router,
    prefix="/predict/carrera",
    tags=["Carrera"],
)

# Calificador Objetivo
# @app.get("/predict/objetivo/")
# def read_objetivo():
#     objetivo = "Aumentar la satisfacción del cliente en un 15% para el tercer trimestre de 2025, implementando un nuevo sistema de soporte en línea y capacitando al equipo de atención al cliente."
#     print(f"Objective text: {objetivo[:100]}...")

#     model_name = "gemma3"
#     approved, verbs, detail, suggestions, suggestion_options = calificate_objective(model_name, objetivo)

#     print(f"Approved: {approved}")
#     print(f"Verbs: {verbs}")
#     print(f"Detail: {detail}")
#     print(f"Suggestions: {suggestions}")
#     print(f"Suggestion Options: {suggestion_options}")

#     return {"model_name": model_name, "approved": approved, "verbs": verbs, "detail": detail, "suggestions": suggestions, "suggestion_options": suggestion_options}

@app.post("/predict/objetivo/", response_model=PredictionResponseClassificationObjective)
def predict_objetivo(item: ItemModelContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    approved, verbs, detail, suggestions, suggestion_options = calificate_objective(model_name, objetivo)

    print(f"Approved: {approved}")
    print(f"Verbs: {verbs}")
    print(f"Detail: {detail}")
    print(f"Suggestions: {suggestions}")
    print(f"Suggestion Options: {suggestion_options}")
    return {"approved": approved, "verbs": verbs, "detail": detail, "suggestions": suggestions, "suggestion_options": suggestion_options}

@app.post("/predict/objetivo/{model_name}", response_model=PredictionResponseClassificationObjective)
def predict_objetivo(model_name: str, item: ItemContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    approved, verbs, detail, suggestions, suggestion_options = calificate_objective(model_name, objetivo)

    print(f"Approved: {approved}")
    print(f"Verbs: {verbs}")
    print(f"Detail: {detail}")
    print(f"Suggestions: {suggestions}")
    print(f"Suggestion Options: {suggestion_options}")
    return {"approved": approved, "verbs": verbs, "detail": detail, "suggestions": suggestions, "suggestion_options": suggestion_options}



# # Calificador Objetivos
# @app.get("/predict/objetivos/")
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

@app.post("/predict/objetivos/", response_model=FullEvaluationResponse)
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

    alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(model_name, objetivo, objetivos_especificos)

    print(f"Approved: {alineacion_aprobada}")
    print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
    print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
    print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")

    response_data = {
        "joint_evaluation": evaluacion_conjunta,
        "individual_evaluation": evaluacion_individual
    }
    
    return response_data

@app.post("/predict/objetivos/{model_name}", response_model=FullEvaluationResponse)
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

    alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(model_name, objetivo, objetivos_especificos)

    print(f"Approved: {alineacion_aprobada}")
    print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
    print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
    print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")
    
    response_data = {
        "joint_evaluation": evaluacion_conjunta,
        "individual_evaluation": evaluacion_individual
    }
    
    return response_data

# --- ENDPOINT MODIFICADO ---
# Este endpoint ahora INICIA la tarea y responde inmediatamente.
@app.post("/predict/objetivos_async/", response_model=TaskCreationResponse, status_code=202)
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

    # 3. Responder inmediatamente con el ID de la tarea
    return {"task_id": task.id, "status": "Processing"}

# --- NUEVO ENDPOINT: Consultar estado de la tarea ---
@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Consulta el estado de una tarea de fondo."""
    task_result = AsyncResult(task_id, app=celery_app)
    
    response_data = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.successful():
        response_data["result"] = task_result.get()
    elif task_result.failed():
        # Si falló, puedes optar por devolver el error
        response_data["result"] = str(task_result.info) # 'info' contiene la excepción

    return response_data

# --- NUEVO ENDPOINT: Obtener resultado de la tarea (alternativa) ---
@app.get("/tasks/{task_id}/result", response_model=FullEvaluationResponse)
def get_task_result(task_id: str):
    """Obtiene el resultado de una tarea completada."""
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready():
        raise HTTPException(status_code=202, detail="La tarea aún no ha finalizado.")
    
    if task_result.failed():
        raise HTTPException(status_code=500, detail=f"La tarea falló: {task_result.info}")

    return task_result.get()