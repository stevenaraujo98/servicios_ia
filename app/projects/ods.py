import os
import numpy as np
from typing import Union
from fastapi import HTTPException, APIRouter
from ..modelsEntity import ItemContent, ItemModelContent, PredictionResponseODS
from ..validations import validate_min_length, validate_not_empty, clean_text
from ..models.ModelLoader import ModelLoader, crear_corpus, detect_language_and_translate_es_en

loader_ods = ModelLoader()

print("Cargando modelos de ods tradicionales...")
loader_ods.load_transformer_model("distilbert_10e_24b_0") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos de ods transformadores...")

# print("Cargando modelos de ods tradicionales...")
# loader_ods.load_traditional_model("Logistic_Regression_20250611_165546") # Cashear el modelo para evitar recargas innecesarias
# print("Finalizó carga de modelos de ods tradicionales...")

ods_router = APIRouter()

def predict_ods_text(model_loader, model_folder, text, model_type='auto'):
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
        predictions = [predictions[0] - 1] # ya que solo es uno
    else:
        predictions, probabilities = model_loader.predict_transformer(model_folder, texts)
    
    prediction = predictions[0] + 1
    probability = probabilities[0][predictions[0]] if probabilities is not None else None

    # Mostrar resultados
    print("\\n📋 Resultados:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        prob_str = ""
        if probabilities is not None:
            max_prob = np.max(probabilities[i])
            prob_str = f" (confianza: {max_prob:.3f})"

        print(f"   {i+1}. Texto: '{text[:75]}...'")
        if model_type == 'traditional':
            print(f"      Predicción: {pred}{prob_str}")
        else:
            print(f"      Predicción: {pred + 1}{prob_str}")
    
    # Obtener el top 3 de probabilidades y sus índices
    top3_indices = []
    top3_probs = []
    if probabilities is not None:
        for prob_list in probabilities:
            indices = np.argsort(prob_list)[-3:][::-1] # Ordena de menor a mayor, con el top 3 al final y -1 mayor a menor
            top3_indices.append(indices.tolist()) # Convertir a lista la lista de índices
            top3_probs.append([prob_list[i] for i in indices])  # obtener las probabilidades correspondientes
    
    return int(prediction), float(probability), predictions, probabilities, top3_indices, top3_probs

# ODS
# @app.get("/predict/ods/")
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

