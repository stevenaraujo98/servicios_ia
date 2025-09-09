from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Union, Dict
import asyncio
import json

# --- Projects ---
from .projects.ods import ods_router
from .projects.patente import patente_router
from .projects.carrera import carrera_router
from .projects.objetivo import objetivo_router
from .projects.objetivos_gen_spe import calificate_objectives_gen_esp

# --- Importaciones de tu proyecto ---
from .validations import validate_min_length, validate_not_empty, clean_text
from .modelsEntity import ItemContentObjectives, ItemModelContentObjectives, FullEvaluationResponse, TaskCreationResponse, TaskStatusResponse
from .models.ModelLoader import ModelLoader
from .redis import ConnectionManager

# --- Imports de Celery y Redis ---
from celery.result import AsyncResult
import redis.asyncio as redis # Usamos la versión asíncrona para FastAPI

# --- Importaciones de Celery tasks ---
from .tasks import celery_app, run_objective_evaluation_task

# =================================================================
# --- SECCIÓN DE WEBSOCKETS Y NOTIFICACIONES EN TIEMPO REAL ---
# =================================================================
manager = ConnectionManager()
# Un diccionario para mapear qué cliente está suscrito a qué tarea.
client_task_map: Dict[str, str] = {}

# --- Listener de Redis ---
async def redis_listener(pubsub):
    print("[DIAGNÓSTICO] Listener de Redis iniciado y escuchando el canal 'task_results'...")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                task_id = data.get("task_id")
                print(f"[DIAGNÓSTICO] Mensaje recibido de Redis para la tarea: {task_id}")

                client_id_to_notify = None
                for cid, tid in client_task_map.items():
                    if tid == task_id:
                        client_id_to_notify = cid
                        break
                
                if client_id_to_notify:
                    print(f"[DIAGNÓSTICO] Intentando enviar resultado al cliente {client_id_to_notify}...")
                    await manager.send_personal_message(json.dumps(data), client_id_to_notify)
                    print(f"[DIAGNÓSTICO] Resultado enviado exitosamente.")
                else:
                    print(f"[DIAGNÓSTICO] No se encontró cliente suscrito a la tarea {task_id}.")
    except asyncio.CancelledError:
        print("[DIAGNÓSTICO] Listener de Redis cancelado.")
    except Exception as e:
        print(f"[DIAGNÓSTICO ERROR] El listener de Redis falló: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[DIAGNÓSTICO] Iniciando lifespan de la aplicación...")
    redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("task_results")
    listener_task = asyncio.create_task(redis_listener(pubsub))
    print("[DIAGNÓSTICO] Tarea de listener de Redis creada.")
    
    yield
    
    print("[DIAGNÓSTICO] Apagando lifespan de la aplicación...")
    listener_task.cancel()
    await pubsub.close()
    await redis_client.close()
    print("[DIAGNÓSTICO] Listener y cliente de Redis cerrados correctamente.")

app = FastAPI(title="API de Modelos IA", description="Endpoints para interactuar con los modelos de IA y Celery.", lifespan=lifespan)

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

app.include_router(
    objetivo_router,
    prefix="/predict/objetivo",
    tags=["Objetivo"],
)

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

# Tarda mucho la respuesta por eso objetivos_async
# @app.post("/predict/objetivos/", response_model=FullEvaluationResponse)
# def predict_objetivos(item: ItemModelContentObjectives, q: Union[str, None] = None):
#     if q:
#         print(f"Query parameter q: {q}")

#     model_name = item.model_name.strip()
#     validate_not_empty(model_name)

#     objetivo = clean_text(item.content)
#     # validate objetivo min limit_min
#     validate_min_length(objetivo, min_length=10)

#     objetivos_especificos = item.specific_objectives
#     if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
#         raise ValueError("La lista de objetivos específicos no puede estar vacía, tampoco menos de 3 ni más de 4.")

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

# Tarda mucho la respuesta por eso objetivos_async
# @app.post("/predict/objetivos/{model_name}", response_model=FullEvaluationResponse)
# def predict_objetivos(model_name: str, item: ItemContentObjectives, q: Union[str, None] = None):
#     if q:
#         print(f"Query parameter q: {q}")
#     validate_not_empty(model_name)

#     objetivo = clean_text(item.content)
#     # validate objetivo min limit_min
#     validate_min_length(objetivo, min_length=10)

#     objetivos_especificos = item.specific_objectives
#     if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
#         raise ValueError("La lista de objetivos específicos no puede estar vacía, tampoco menos de 3 ni más de 4.")

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

@app.on_event("startup")
async def startup_event():
    # Inicia el listener de Redis cuando la app arranca.
    asyncio.create_task(redis_listener(manager))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint que maneja las conexiones WebSocket."""
    await manager.connect(websocket)
    try:
        task_id = await websocket.receive_text()
        manager.associate_task_id(task_id, websocket)
        # Mantenemos la conexión abierta esperando la notificación del listener.
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect_by_websocket(websocket)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "subscribe":
                task_id = message.get("task_id")
                if task_id:
                    client_task_map[client_id] = task_id
                    print(f"[DIAGNÓSTICO] Cliente {client_id} suscrito a la tarea: {task_id}")
                    await manager.send_personal_message(f"Suscrito exitosamente a la tarea {task_id}", client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        if client_id in client_task_map:
            del client_task_map[client_id]

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

# --- Consultar estado de la tarea ---
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

# --- Obtener resultado de la tarea (alternativa) ---
@app.get("/tasks/{task_id}/result", response_model=FullEvaluationResponse)
def get_task_result(task_id: str):
    """Obtiene el resultado de una tarea completada."""
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready():
        raise HTTPException(status_code=202, detail="La tarea aún no ha finalizado.")
    
    if task_result.failed():
        raise HTTPException(status_code=500, detail=f"La tarea falló: {task_result.info}")

    return task_result.get()

# --- Endpoint de Salud para el Healthcheck de Docker ---
@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}
