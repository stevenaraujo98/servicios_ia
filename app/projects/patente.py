import os
import numpy as np
from typing import Union
from fastapi import HTTPException, APIRouter
from ..modelsEntity import ItemContent, ItemModelContent, PredictionResponse
from ..validations import validate_min_length, validate_not_empty, clean_text
from ..models.ModelLoader import ModelLoader, crear_corpus, detect_language_and_translate_es_en

loader_patente = ModelLoader(tipo='patente')

# print("Cargando modelos de patente tradicionales...")
# loader_patente.load_transformer_model("distilbert_20250710_0233") # Cashear el modelo para evitar recargas innecesarias
# print("Finalizó carga de modelos de patente transformadores...")

print("Cargando modelos de patente tradicionales...")
loader_patente.load_traditional_model("Random_Forest_20250813_144340") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos de patente tradicionales...")

patente_router = APIRouter()

def predict_patent_text(model_loader, model_folder, text, model_type='auto'):
    """Predice etiquetas para textos individuales"""
    print(f"Procesamiento de texto para predicción con modelo: {model_folder}")
    print(f"   - Para el texto: {text[:75]}...")

    # Detectar tipo de modelo
    if model_type == 'auto':
        if model_folder in model_loader.loaded_models:
            model_type = model_loader.loaded_models[model_folder]['type']
        else:
            print("🔍 Detectando tipo de modelo...")
            if os.path.exists(f"{model_loader.models_dir}/traditional/{model_folder}"):
                print("   - Modelo tradicional detectado", model_loader.models_dir+"/traditional/"+model_folder)
                model_type = 'traditional'
            elif os.path.exists(f"{model_loader.models_dir}/transformers/{model_folder}"):
                print("   - Modelo transformer detectado", model_loader.models_dir+"/transformers/"+model_folder)
                model_type = 'transformer'
            else:
                raise HTTPException(status_code=404, detail=f"Modelo {model_folder} no encontrado.")

    # Lematizar y limpiar textos
    texts = [crear_corpus(text)]
    texts = detect_language_and_translate_es_en(texts) # Detectar idioma y traducir a ingles en caso este en español

    print(f"\\n🔮 Prediciendo {len(texts)} textos con modelo: {model_folder}")

    print(f" Ejecutando predicción ...")
    # Hacer prediccion del texto
    if model_type == 'traditional':
        predictions, probabilities = model_loader.predict_traditional(model_folder, texts)
        predictions = [predictions[0]] # ya que solo es uno
    else:
        predictions, probabilities = model_loader.predict_transformer(model_folder, texts)

    prediction = predictions[0]
    probability = probabilities[0][predictions[0]] if probabilities is not None else None

    # Mostrar resultados
    print("\\n📋 Resultados:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        prob_str = ""
        if probabilities is not None:
            max_prob = np.max(probabilities[i])
            prob_str = f" (confianza: {max_prob:.3f})"

        print(f"   {i+1}. Texto: '{text[:75]}...'")
        print(f"   Predicción: {pred}{prob_str}")

    return int(prediction), round(float(probability), 2) if probability is not None else None, predictions, probabilities

# @app.get("/predict/patente/")
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
