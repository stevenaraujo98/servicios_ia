from fastapi import APIRouter, HTTPException

# Importamos la tarea de Celery
from app.celery.tasks import run_analisis_sentimiento_task

# --- Importaciones de tu proyecto ---
from app.consts import stages
from app.entities import ItemContent, TaskCreationResponse, SentimentResponse
from app.validations import validate_min_length, clean_text, validation_response_redis

# --- Importaciones de Celery tasks ---
from app.celery.tasks import celery_app
from celery.result import AsyncResult

router_sentimiento = APIRouter()

@router_sentimiento.post("/async", response_model=TaskCreationResponse, status_code=202)
def predict_sentimiento_async(item: ItemContent):
    texto_a_analizar = clean_text(item.content)
    validate_min_length(texto_a_analizar, min_length=10)

    # Iniciamos la nueva tarea en segundo plano
    task = run_analisis_sentimiento_task.delay(texto_a_analizar)

    # Respondemos inmediatamente con el ID de la tarea
    return {"task_id": task.id, "status": stages[0]}

# --- Obtener resultado de la tarea ---
@router_sentimiento.get("/result/{task_id}", response_model=SentimentResponse)
def get_task_result(task_id: str):
    """Obtiene el resultado de una tarea completada."""
    task_result = AsyncResult(task_id, app=celery_app)

    return validation_response_redis(task_result, SentimentResponse)
