from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Importamos la tarea de Celery que crearemos en el siguiente paso
from app.celery.tasks import run_analisis_sentimiento_task
from app.consts import stages
from app.entities import ItemContent, TaskCreationResponse

# --- Importaciones de Celery tasks ---
from app.celery.tasks import celery_app
from celery.result import AsyncResult

router_sentimiento = APIRouter()

@router_sentimiento.post("/async", response_model=TaskCreationResponse, status_code=202)
def predict_sentimiento_async(item: ItemContent):
    if not item.texto_a_analizar or len(item.texto_a_analizar) < 5:
        raise HTTPException(status_code=400, detail="El texto es demasiado corto.")

    # Iniciamos la nueva tarea en segundo plano
    task = run_analisis_sentimiento_task.delay(item.texto_a_analizar)

    # Respondemos inmediatamente con el ID de la tarea
    return {"task_id": task.id, "status": stages[0]}

# --- Obtener resultado de la tarea ---
@router_sentimiento.get("/result/{task_id}")#, response_model=FullEvaluationResponse)
def get_task_result(task_id: str):
    """Obtiene el resultado de una tarea completada."""
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready():
        raise HTTPException(status_code=202, detail="La tarea aún no ha finalizado.")
    
    if task_result.failed():
        raise HTTPException(status_code=500, detail=f"La tarea falló: {task_result.info}")

    return task_result.get()
