from typing import Union
from fastapi import APIRouter

from .logic import predict_carrera_text
# --- Importaciones de tu proyecto ---
from app.models.ModelLoader import ModelLoader
from app.entities import ItemContent, ItemModelContent, PredictionResponseCareer
from app.validations import validate_min_length, validate_not_empty, clean_text

loader_carrera = ModelLoader(tipo='carrera')

# print("Cargando modelos de carrera transformadores...")
# 1.2.2 de kaggle
# loader_carrera.load_transformer_model("bert_20250806_234119") # Cashear el modelo para evitar recargas innecesarias
# print("Finalizó carga de modelos de carrera transformadores...")

print("Cargando modelos de carrera tradicionales...")
loader_carrera.load_traditional_model("Random_Forest_20250808_161322") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos de carrera tradicionales...")

carrera_router = APIRouter()

# CARRERA
# @app.get("/")
# def read_carrera():
#     sample_text = "Desarrollo de un prototipo de sistema para el seguimiento de contratos para la espol." # Computación
#     print(f"Sample text: {sample_text[:100]}...")

#     model_name = "Random_Forest_20250804_100503"
#     prediction, probability, class_label, probabilities, top3_career, top3_probs = predict_carrera_text(loader_carrera, model_name, sample_text)

#     print(f"Prediction: {prediction} with probability: {probability}")
#     print(f"Clases: {type(class_label)}")
#     print(f"Clases: {class_label}")
#     print(f"Probabilities: {probabilities}")
#     print(f"Top 3 Indices: {top3_career}")
#     print(f"Top 3 Probabilities: {top3_probs}")

#     return {"model_name": model_name, "prediction": prediction, "probability": probability, "top3_career": top3_career, "top3_probs": top3_probs, "class_label": class_label}

@carrera_router.post("/", response_model=PredictionResponseCareer)
def predict_carrera(item: ItemModelContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    sample_text = clean_text(item.content)
    # validate sample_text min limit_min
    validate_min_length(sample_text, min_length=10)

    prediction, probability, class_label, probabilities, top3_careers, top3_probs = predict_carrera_text(loader_carrera, model_name, sample_text)
    print(f"Prediction: {prediction} with probability: {probability}")
    print(f"Probabilities: {len(probabilities)}")
    print(f"class_label: {len(class_label)}")
    print(f"Top 3 Carreras: {top3_careers}")
    print(f"Top 3 Probabilities: {top3_probs}")
    return {"prediction": prediction, "probability": probability, "top3_careers": top3_careers, "top3_probabilities": top3_probs}

@carrera_router.post("/{model_name}", response_model=PredictionResponseCareer)
def predict_carrera(model_name: str, item: ItemContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")
    validate_not_empty(model_name)

    sample_text = clean_text(item.content)
    # validate sample_text min limit_min
    validate_min_length(sample_text, min_length=10)

    prediction, probability, class_label, probabilities, top3_careers, top3_probs = predict_carrera_text(loader_carrera, model_name, sample_text)
    print(f"Prediction: {prediction} with probability: {probability}")
    print(f"Probabilities: {len(probabilities)}")
    print(f"class_label: {len(class_label)}")
    print(f"Top 3 Carreras: {top3_careers}")
    print(f"Top 3 Probabilities: {top3_probs}")
    return {"prediction": prediction, "probability": probability, "top3_careers": top3_careers, "top3_probabilities": top3_probs}
