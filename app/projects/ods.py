import numpy as np
import os
from ..models.ModelLoader import crear_corpus, detect_language_and_translate_es_en
from fastapi import HTTPException

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