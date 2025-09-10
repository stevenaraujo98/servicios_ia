from typing import Union
from fastapi import  APIRouter

from .logic import predict_patent_text
# --- Importaciones de tu proyecto ---
from app.entities import ItemContent, ItemModelContent, PredictionResponse
from app.validations import validate_min_length, validate_not_empty, clean_text
from app.models.ModelLoader import ModelLoader

loader_patente = ModelLoader(tipo='patente')

# print("Cargando modelos de patente tradicionales...")
# loader_patente.load_transformer_model("distilbert_20250710_0233") # Cashear el modelo para evitar recargas innecesarias
# print("Finalizó carga de modelos de patente transformadores...")

print("Cargando modelos de patente tradicionales...")
loader_patente.load_traditional_model("Random_Forest_20250813_144340") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos de patente tradicionales...")

patente_router = APIRouter()

# @app.get("/")
# def read_project():
#     sample_text = "METHOD FOR MANAGING INDOOR BEACON-BASED COMMUNICATION A method for content distribution in an indoor space and surrounding area covered by beacon signals. The method includes setting general data of the content, setting at least one location in the indoor space, including one or a combination of active/inactive locations and other relevant information available in a system, assigning beacons that trigger the content in the indoor space, receiving by the system, from a portable device via a first communication channel, beacon information that is received from a beacon by the portable device via a second communication channel, the first communication channel being different from the second communication channel, setting a singular event or plural events which initiates the content in the indoor space, setting a condition formula for the content which must be fulfilled to qualify for the content, and setting singular or plural results of the content that are provided when the condition formula is fulfilled."
#     print(f"Sample text: {sample_text[:100]}...")

#     model_name = "Random_Forest_20250708_144028"
#     prediction, probability, predictions, probabilities = predict_patent_text(loader_patente, model_name, sample_text)

#     print(f"Prediction: {prediction} with probability: {probability}")
#     print(f"Predictions: {predictions}")
#     print(f"Probabilities: {probabilities}")

#     return {"model_name": model_name, "prediction": prediction, "probability": probability, "predictions": predictions, "probabilities": probabilities}

@patente_router.post("/", response_model=PredictionResponse)
def predict_project(item: ItemModelContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    sample_text = clean_text(item.content)
    # validate sample_text min limit_min
    validate_min_length(sample_text)

    prediction, probability, predictions, probabilities = predict_patent_text(loader_patente, model_name, sample_text)
    probabilities = probabilities[0] if len(probabilities) > 0 else None  # Asegurar que las probabilidades sean una lista
    print(f"Prediction: {prediction} with probability: {probability}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")
    return {"prediction": prediction, "probability": probability, "predictions": predictions, "probabilities": probabilities}

@patente_router.post("/{model_name}", response_model=PredictionResponse)
def predict_project(model_name: str, item: ItemContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    validate_not_empty(model_name)

    sample_text = clean_text(item.content)
    # validate sample_text min limit_min
    validate_min_length(sample_text)

    prediction, probability, predictions, probabilities = predict_patent_text(loader_patente, model_name, sample_text)
    probabilities = probabilities[0] if len(probabilities) > 0 else None  # Asegurar que las probabilidades sean una lista
    print(f"Prediction: {prediction} with probability: {probability}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")
    return {"prediction": prediction, "probability": probability, "predictions": predictions, "probabilities": probabilities}
