from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict
import asyncio
import json
from .consts import tags_metadata, REDIS_HOST, REDIS_PORT # , REDIS_STORE_DB_INDEX

# --- Projects ---
from .projects.ods.router import ods_router
from .projects.patente.router import patente_router
from .projects.carrera.router import carrera_router
from .projects.objetivos.router import objetivo_router
from .projects.analisis_sentimiento.router import router_sentimiento
from .projects.objetivos_gen_spec.router import objetivo_gen_spe_router

# --- Imports de Celery y Redis ---
from .redis import ConnectionManager
import redis.asyncio as redis # Versión asíncrona para FastAPI

from .entities import TaskStatusResponse 

# --- Importaciones de Celery tasks ---
from .celery.tasks import celery_app
from celery.result import AsyncResult

# =================================================================
# --- SECCIÓN DE WEBSOCKETS Y NOTIFICACIONES EN TIEMPO REAL ---
# =================================================================
manager = ConnectionManager()
# Un diccionario para mapear qué cliente está suscrito a qué tarea.
client_task_map: Dict[str, str] = {}

# --- Listener de Redis ---
async def redis_listener(pubsub):
    print("[DIAGNÓSTICO] Listener de Redis iniciado...")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                # Ahora el mensaje es un JSON puro, lo decodificamos directamente.
                data = json.loads(message["data"])
                task_id = data.get("task_id")
                print(f"[DIAGNÓSTICO] Mensaje recibido de Redis para la tarea: {task_id}")

                client_id_to_notify = None
                for cid, tid in client_task_map.items():
                    if tid == task_id:
                        client_id_to_notify = cid
                        break
                
                if client_id_to_notify:
                    print(f"[DIAGNÓSTICO] Enviando resultado al cliente {client_id_to_notify}...")
                    # Enviamos el JSON completo al cliente
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
    # Usamos las constantes para la conexión
    # si no se especifica un número de base de dato, por defecto es 0 (db=REDIS_STORE_DB_INDEX)
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
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

app = FastAPI(
    title="API de Modelos IA", 
    description="REST API y WebSocket para interactuar con los modelos de IA.", 
    lifespan=lifespan, 
    version="1.1.0",
    openapi_tags=tags_metadata
)

# --- Middleware CORS ---
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

@app.get("/", tags=["Default"], status_code=200)
def read_root():
    return {"Hello": "IA", "status": "ok"}

# Patente
app.include_router(patente_router, prefix="/predict/patente", tags=["Patente"])

# ODS
app.include_router(ods_router, prefix="/predict/ods", tags=["ODS"])

# Carrera
app.include_router(carrera_router, prefix="/predict/carrera", tags=["Carrera"])

# Objetivo
app.include_router(objetivo_router, prefix="/predict/objetivo", tags=["Objetivo"])

# Objetivos general y especifico
app.include_router(objetivo_gen_spe_router, prefix="/predict/objetivos", tags=["Objetivos: general y específicos"])

# Analisis de sentimiento
app.include_router(router_sentimiento, prefix="/predict/sentimiento", tags=["Análisis de Sentimiento"])

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

# --- Consultar estado de cualquier tarea --- 
# tags auto-generado llamado "default"
@app.get("/predict/status/{task_id}", tags=["Default"], response_model=TaskStatusResponse)
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
