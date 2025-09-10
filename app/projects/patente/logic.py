import os
import numpy as np
from fastapi import HTTPException

# --- Importaciones de tu proyecto ---
from app.models.ModelLoader import crear_corpus, detect_language_and_translate_es_en

def predict_patent_text(loader_patente, model_folder, text, model_type='auto'):
    """Predice etiquetas para textos individuales"""
    print(f"Procesamiento de texto para predicción con modelo: {model_folder}")
    print(f"   - Para el texto: {text[:75]}...")

    # Detectar tipo de modelo
    if model_type == 'auto':
        if model_folder in loader_patente.loaded_models:
            model_type = loader_patente.loaded_models[model_folder]['type']
        else:
            print("🔍 Detectando tipo de modelo...")
            if os.path.exists(f"{loader_patente.models_dir}/traditional/{model_folder}"):
                print("   - Modelo tradicional detectado", loader_patente.models_dir+"/traditional/"+model_folder)
                model_type = 'traditional'
            elif os.path.exists(f"{loader_patente.models_dir}/transformers/{model_folder}"):
                print("   - Modelo transformer detectado", loader_patente.models_dir+"/transformers/"+model_folder)
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
        predictions, probabilities = loader_patente.predict_traditional(model_folder, texts)
        predictions = [predictions[0]] # ya que solo es uno
    else:
        predictions, probabilities = loader_patente.predict_transformer(model_folder, texts)

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
