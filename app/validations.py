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
    if len(text) < min_length:
        raise HTTPException(status_code=422, detail=f"El texto debe tener mínimo {min_length} caracteres.")

def validate_not_empty(text: str):
    if not text.strip():
        raise HTTPException(status_code=422, detail="El texto no puede estar vacío o contener solo espacios en blanco.")