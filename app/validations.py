from pydantic import ValidationError
from fastapi import HTTPException
from .consts import limit_min
import string

def clean_text(text: str) -> str:
    """Limpia el texto eliminando espacios extra y puntuación innecesaria."""
    if text is None:
        return ""
    # Eliminar múltiples espacios en blanco y dejar solo uno
    cleaned_text = ' '.join(text.split())
    return cleaned_text.translate(str.maketrans('', '', string.punctuation))

def validate_min_length(text: str, min_length: int = limit_min):
    """Verifica el tamaño minimo del texto"""
    if len(text) < min_length:
        raise HTTPException(status_code=422, detail=f"El texto debe tener mínimo {min_length} caracteres.")

def validate_not_empty(text: str):
    """Verifica que el texto no este vacio"""
    if not text.strip():
        raise HTTPException(status_code=422, detail="El texto no puede estar vacío o contener solo espacios en blanco.")
    
def validation_response_redis(task_result, model_struc):
    """Valida si la respuesta esta aun ejecutandose, si no esta autorizado el token, si ha fallado la tarea, si el token tiene la estructura que se espera responder ya que puede ser incorrecto el token"""
    if not task_result.ready():
        raise HTTPException(status_code=202, detail="La tarea aún no ha finalizado.")
    
    if task_result.failed():
        error_info = str(task_result.info)
        if "token" in error_info.lower() or "unauthorized" in error_info.lower():
            raise HTTPException(
                status_code=401, 
                detail="Token de autenticación inválido o expirado"
            )
        raise HTTPException(status_code=500, detail=f"La tarea falló: {error_info}")

    result_data = task_result.get()
    
    try:
        validated_response = model_struc(**result_data)
        return validated_response
    except ValidationError as e:
        raise HTTPException(
            status_code=422, 
            detail="El token proporcionado no es válido o la estructura de respuesta es incorrecta"
        )
