import numpy as np
import os
from ..models.ModelLoader import crear_corpus, detect_language_and_translate_en_es
from fastapi import HTTPException

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