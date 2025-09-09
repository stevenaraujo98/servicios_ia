import os
import numpy as np
from typing import Union
from fastapi import HTTPException, APIRouter

# --- Importaciones de tu proyecto ---
from ..models.ModelLoader import ModelLoader, crear_corpus
from ..modelsEntity import ItemContent, ItemModelContent, PredictionResponseCareer
from ..validations import validate_min_length, validate_not_empty, clean_text

loader_carrera = ModelLoader(tipo='carrera')

# print("Cargando modelos de carrera transformadores...")
# 1.2.2 de kaggle
# loader_carrera.load_transformer_model("bert_20250806_234119") # Cashear el modelo para evitar recargas innecesarias
# print("Finalizó carga de modelos de carrera transformadores...")

print("Cargando modelos de carrera tradicionales...")
loader_carrera.load_traditional_model("Random_Forest_20250808_161322") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos de carrera tradicionales...")

carrera_router = APIRouter()

def predict_carrera_text(model_loader, model_folder, text, model_type='auto'):
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
    texts = detect_language_and_translate_en_es(texts) # Detectar idioma y traducir a español en caso este en ingles

    print(f"\\n🔮 Prediciendo {len(texts)} textos con modelo: {model_folder}")

    print(f" Ejecutando predicción ...")
    # Hacer prediccion del texto
    if model_type == 'traditional':
        predictions, probabilities, label_predictions = model_loader.predict_traditional(model_folder, texts)
    else:
        predictions, probabilities, label_predictions = model_loader.predict_transformer(model_folder, texts)
    
    prediction = label_predictions[predictions[0]] # texto de etiqueta y es un solo valor de predictions
    probability = probabilities[0][predictions[0]] if probabilities is not None else None

    # Mostrar resultados
    print("\\n📋 Resultados:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        prob_str = ""
        if probabilities is not None:
            max_prob = np.max(probabilities[i])
            prob_str = f" (confianza: {max_prob:.3f})"

        print(f"   {i+1}. Texto: '{text[:75]}...'")
        print(f"      Predicción: {pred}{prob_str} - Clase: {label_predictions[pred]}")
    
    # Obtener el top 3 de probabilidades y sus índices
    top3_career = []
    top3_probs = []
    if probabilities is not None:
        prob_list = probabilities[0]
        indices = np.argsort(prob_list)[-3:][::-1] # Ordena de menor a mayor, con el top 3 al final y -1 mayor a menor
        top3_careers = [label_predictions[i] for i in indices] # Convertir a lista la lista de índices
        top3_probs = [prob_list[i] for i in indices]  # obtener las probabilidades correspondientes

    return prediction, float(probability), label_predictions, probabilities, top3_careers, top3_probs

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


