import json
from ollama import chat
from typing import Union
from fastapi import APIRouter

from .logic import calificate_objective
# --- Importaciones de tu proyecto ---
from app.validations import validate_min_length, validate_not_empty, clean_text
from app.entities import ItemContent, ItemModelContent, PredictionResponseClassificationObjective

objetivo_router = APIRouter()

# Calificador Objetivo
# @app.get("/")
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

@objetivo_router.post("/", response_model=PredictionResponseClassificationObjective)
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

@objetivo_router.post("/{model_name}", response_model=PredictionResponseClassificationObjective)
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
