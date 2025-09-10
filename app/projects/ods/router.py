from typing import Union
from fastapi import APIRouter

from .logic import predict_ods_text
# --- Importaciones de tu proyecto ---
from app.validations import validate_min_length, validate_not_empty, clean_text
from app.entities import ItemContent, ItemModelContent, PredictionResponseODS
from app.models.ModelLoader import ModelLoader

loader_ods = ModelLoader()

print("Cargando modelos de ods tradicionales...")
loader_ods.load_transformer_model("distilbert_10e_24b_0") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos de ods transformadores...")

# print("Cargando modelos de ods tradicionales...")
# loader_ods.load_traditional_model("Logistic_Regression_20250611_165546") # Cashear el modelo para evitar recargas innecesarias
# print("Finalizó carga de modelos de ods tradicionales...")

ods_router = APIRouter()

# ODS
# @app.get("/")
# def read_text():
#     sample_text = "Food waste and food insecurity are pressing global challenges. This study presents a novel approach to optimizing the food bank network redesign (FBNR) by leveraging the Quito Metro system to create a decentralized food bank network. We propose positioning lockers at metro stations for convenient food donations, which are then transported using the metro’s spare capacity to designated stations for collection by charities. A blockchain-based traceability system with smart contracts serves as the core data management system, ensuring secure and transparent traceability of donations. Additionally, we develop a multi-objective optimization model aiming to minimize food waste, reduce transportation costs, and increase the social impact of food distribution. A mixed-integer linear programming (MIP) model further optimizes the allocation of donations to ensure efficient distribution. By integrating these models with the blockchain system, we offer a comprehensive solution to the FBNR, promoting a more sustainable and equitable food system."
#     print(f"Sample text: {sample_text[:100]}...")
#
#     # model_name = "SVM_20250611_165538"
#     model_name = "distilbert_10e_24b_0"
#     prediction, probability, predictions, probabilities, top3_indices, top3_probs = predict_ods_text(loader_ods, model_name, sample_text)

#     print(f"Prediction: {prediction} with probability: {probability}")
#     print(f"Predictions: {predictions}")
#     print(f"Probabilities: {probabilities}")

#     return {"model_name": model_name, "prediction": prediction, "probability": probability, "predictions": predictions, "probabilities": probabilities}

@ods_router.post("/", response_model=PredictionResponseODS)
def predict_text(item: ItemModelContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    sample_text = clean_text(item.content)
    # validate sample_text min limit_min
    validate_min_length(sample_text)

    prediction, probability, predictions, probabilities, top3_indices, top3_probs = predict_ods_text(loader_ods, model_name, sample_text)
    print(f"Prediction: {prediction} with probability: {probability}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")
    print(f"Top 3 Indices: {top3_indices}")
    print(f"Top 3 Probabilities: {top3_probs}")
    return {"prediction": prediction, "probability": probability, "predictions": top3_indices, "probabilities": probabilities}

@ods_router.post("/{model_name}", response_model=PredictionResponseODS)
def predict_text(model_name: str, item: ItemContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    validate_not_empty(model_name)

    sample_text = clean_text(item.content)
    # validate sample_text min limit_min
    validate_min_length(sample_text)

    prediction, probability, predictions, probabilities, top3_indices, top3_probs = predict_ods_text(loader_ods, model_name, sample_text)
    print(f"Prediction: {prediction} with probability: {probability}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")
    print(f"Top 3 Indices: {top3_indices}")
    print(f"Top 3 Probabilities: {top3_probs}")
    return {"prediction": prediction, "probability": probability, "predictions": top3_indices, "probabilities": probabilities}

