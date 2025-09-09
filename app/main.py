from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict
import asyncio
import json
from consts import REDIS_PORT

# --- Projects ---
from .projects.ods import ods_router
from .projects.patente import patente_router
from .projects.carrera import carrera_router
from .projects.objetivo import objetivo_router
from .projects.objetivos_gen_spe import objetivo_gen_spe_router

# --- Imports de Celery y Redis ---
from .redis import ConnectionManager
import redis.asyncio as redis # Versión asíncrona para FastAPI


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
    redis_client = redis.Redis(host='redis', port=REDIS_PORT, decode_responses=True)
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
    description="Endpoints para interactuar con los modelos de IA y Celery.", 
    lifespan=lifespan, 
    version="1.0.0"
)

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

@app.get("/", status_code=200)
def read_root():
    return {"Hello": "IA", "status": "ok"}

# Patente
app.include_router(
    patente_router,
    prefix="/predict/patente",
    tags=["Patente"],
)

# ODS
app.include_router(
    ods_router,
    prefix="/predict/ods",
    tags=["ODS"],
)

# Carrera
app.include_router(
    carrera_router,
    prefix="/predict/carrera",
    tags=["Carrera"],
)

# Objetivo
app.include_router(
    objetivo_router,
    prefix="/predict/objetivo",
    tags=["Objetivo"],
)

# Objetivos general y especifico
app.include_router(
    objetivo_gen_spe_router,
    prefix="/predict/objetivos",
    tags=["Objetivos: general y específico"],
)

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
