from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    MarianMTModel, MarianTokenizer
)

from nltk.corpus import stopwords
import joblib
import string
import torch
import spacy
import os
import re

nlp = spacy.load("en_core_web_lg")

stopwords_es = set(stopwords.words('spanish'))
stopwords_en = set(stopwords.words('english'))
stop_words = stopwords_es | stopwords_en

# Detección de idioma
model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer_detected = AutoTokenizer.from_pretrained(model_ckpt)
model_detected = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

# Traducción de español a inglés
model_name_es_en = "Helsinki-NLP/opus-mt-es-en"
tokenizer_tradu_es_en = MarianTokenizer.from_pretrained(model_name_es_en)
model_tradu_es_en = MarianMTModel.from_pretrained(model_name_es_en)

# Traduccion de inglés a español
model_name_en_es = "Helsinki-NLP/opus-mt-en-es"
tokenizer_tradu_en_es = MarianTokenizer.from_pretrained(model_name_en_es)
model_tradu_en_es = MarianMTModel.from_pretrained(model_name_en_es)

# Función de limpieza y lematización
def procesar_texto(texto):
    if texto is None or texto.strip() == "":
        return ""

    # Remove numbers
    texto = re.sub(r'\d+', '', texto)

    # Remove special characters (keep only letters and spaces)
    texto = re.sub(r'\W+', ' ', texto)

    # Unicode, minúsculas, quitar puntuación
    texto = texto.strip().lower().translate(str.maketrans('', '', string.punctuation))

    # Replace multiple spaces with a single space
    texto = re.sub(r'\s+', ' ', texto)

    # Procesar con spaCy
    doc = nlp(texto)

    # Lematizar y quitar stopwords
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_punct and not token.is_space]

    return " ".join(tokens)

# Unir columnas y procesar
def crear_corpus(row):
    return procesar_texto(str(row))


def get_translation_es_en(text):
  translated = model_tradu_es_en.generate(**tokenizer_tradu_es_en(text, return_tensors="pt", padding=True))
  return tokenizer_tradu_es_en.decode(translated[0], skip_special_tokens=True)

def detect_language_and_translate_es_en(list_text):
    result_list = [] # lista de textos con los idiomas
    inputs = tokenizer_detected(list_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model_detected(**inputs).logits

    preds = torch.softmax(logits, dim=-1)
    id2lang = model_detected.config.id2label
    vals, idxs = torch.max(preds, dim=1)
    result_list.extend([(id2lang[k.item()], v.item()) for k, v in zip(idxs, vals)])

    list_new_text = []
    for i in range(len(result_list)):
        lang = result_list[i][0]
        if lang == "es":
            print(f"Translating text {i} from Spanish to English")
            resp = get_translation_es_en(list_text[i])
            list_new_text.append(resp)
        else:
            list_new_text.append(list_text[i])
    
    return list_new_text


def get_translation_en_es(text):
  translated = model_tradu_en_es.generate(**tokenizer_tradu_en_es(text, return_tensors="pt", padding=True))
  return tokenizer_tradu_en_es.decode(translated[0], skip_special_tokens=True)

def detect_language_and_translate_en_es(list_text):
    result_list = [] # lista de textos con los idiomas
    inputs = tokenizer_detected(list_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model_detected(**inputs).logits

    preds = torch.softmax(logits, dim=-1)
    id2lang = model_detected.config.id2label
    vals, idxs = torch.max(preds, dim=1)
    result_list.extend([(id2lang[k.item()], v.item()) for k, v in zip(idxs, vals)])

    list_new_text = []
    for i in range(len(result_list)):
        lang = result_list[i][0]
        if lang == "en":
            print(f"traduciendo el texto {i} de Inglés a Español")
            resp = get_translation_en_es(list_text[i])
            list_new_text.append(resp)
        else:
            list_new_text.append(list_text[i])
    
    return list_new_text


class ModelLoader:
    """Clase para cargar y usar modelos guardados"""
    
    def __init__(self, tipo='ods'):
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), tipo) # ruta absoluta al directorio de modelos
        self.loaded_models = {}
        self.tipo = tipo
        print(f"ModelLoader initialized for type: {self.tipo} with models directory: {self.models_dir}")
  
    def load_traditional_model(self, model_folder):
        """Carga modelo tradicional"""
        model_dir = f"{self.models_dir}/traditional/{model_folder}"

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Modelo no encontrado: {model_dir}")
        
        print(f"📥 Cargando modelo tradicional: {model_folder}")
        
        # Cargar componentes
        model_path = f"{model_dir}/model.pkl"
        vectorizer_path = f"{model_dir}/vectorizer.pkl"
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # try open label enconder
        label_path = f"{model_dir}/label_encoder.pkl"
        label_encoder = None
        try:
            label_encoder = joblib.load(label_path)
        except FileNotFoundError:
            print("No tiene label encoder")
        except Exception as e:
            print(f"Error al cargar label encoder: {e}")
        
        # Guardar en cache
        self.loaded_models[model_folder] = {
            'type': 'traditional',
            'model': model,
            'vectorizer': vectorizer,
            'label_encoder': label_encoder
        }
        
        print(f"   ✅ Modelo cargado exitosamente")
        # return model, vectorizer, label_encoder

    def load_transformer_model(self, model_folder, device=None):
        """Carga modelo transformer"""
        model_dir = f"{self.models_dir}/transformers/{model_folder}"

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Modelo no encontrado: {model_dir}")
        
        print(f"📥 Cargando modelo transformer: {model_folder}")
        
        # Detectar dispositivo
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"   🔧 Usando dispositivo: {device}")        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model = model.to(device)
        model.eval()

        # try open label enconder
        label_path = f"{model_dir}/label_encoder.pkl"
        label_encoder = None
        try:
            label_encoder = joblib.load(label_path)
        except FileNotFoundError:
            print("No tiene label encoder")
        except Exception as e:
            print(f"Error al cargar label encoder: {e}")
        
        # Guardar en cache
        self.loaded_models[model_folder] = {
            'type': 'transformer',
            'model': model,
            'tokenizer': tokenizer,
            'device': device,
            'label_encoder': label_encoder
        }
        
        print(f"   ✅ Modelo cargado exitosamente")
        # return model, tokenizer
    
    '''
    predictions: lista de enteros (índices de clases)
    probabilities: lista de listas de probabilidades
    label_encoder: para convertir índices a nombres de clases (si carrera)
    
    returns: predictions, probabilities, label_encoder.classes_ (if carrera)
    '''
    def predict_traditional(self, model_folder, texts):
        """Predicción con modelo tradicional"""
        print(f"🔍 Cargando modelo tradicional: {model_folder}")
        if model_folder not in self.loaded_models:
            print("Modelo no encontrado en memoria, procediendo a cargarlo...")
            self.load_traditional_model(model_folder)

        model_data = self.loaded_models[model_folder]
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        label_encoder = model_data['label_encoder'] # si no es carrera esta vacio

        print(f"🔍 Procesando {len(texts)} textos para predicción...")
        new_list_lema = []
        for text in texts:
            new_list_lema.append(crear_corpus(text))

        # Vectorizar textos
        X_vec = vectorizer.transform(new_list_lema)
        
        # Predicciones
        predictions = model.predict(X_vec)
        probabilities = None
        
        # Obtener probabilidades si el modelo las soporta
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_vec)

        # Convertir de numpy a listas antes de retornar
        predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
        probabilities_list = probabilities.tolist() if probabilities is not None and hasattr(probabilities, 'tolist') else probabilities

        if self.tipo == "carrera":
            return predictions_list, probabilities_list, label_encoder.classes_.tolist() if label_encoder is not None else None

        return predictions_list, probabilities_list
    
    '''
    predictions: lista de enteros (índices de clases)
    probabilities: lista de listas de probabilidades
    label_encoder: para convertir índices a nombres de clases (si carrera)

    returns: predictions, probabilities, label_encoder.classes_ (if carrera)
    '''
    def predict_transformer(self, model_folder, texts, batch_size=16):
        print(f"🔍 Cargando modelo transformer: {model_folder}")
        """Predicción con modelo transformer"""
        if model_folder not in self.loaded_models:
            print("Modelo no encontrado en memoria, procediendo a cargarlo...")
            self.load_transformer_model(model_folder)

        model_data = self.loaded_models[model_folder]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        device = model_data['device']
        label_encoder = model_data['label_encoder'] # si no es carrera esta vacio

        print(f"🔍 Procesando {len(texts)} textos para predicción...")
        new_list_lema = []
        for text in texts:
            new_list_lema.append(crear_corpus(text))
        
        predictions = []
        probabilities = []

        if self.tipo == "carrera":
            label_encoder
        
        # Procesar en lotes
        for i in range(0, len(new_list_lema), batch_size):
            batch_texts = new_list_lema[i:i+batch_size]
            
            # Tokenizar
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Mover al dispositivo
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predicción
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Probabilidades
                probs = torch.softmax(logits, dim=-1)
                probabilities.extend(probs.cpu().numpy())
                
                # Predicciones
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
        
        # Convertir a lista y entero cada elemento interno de predictions y probabilities  
        tmp_predictions = predictions.copy()
        for i, pred in enumerate(tmp_predictions):
            predictions[i] = int(pred)
            probabilities[i] = probabilities[i].tolist() if probabilities is not None else None


        if self.tipo == "carrera":
            return predictions, probabilities, label_encoder.classes_.tolist() if label_encoder is not None else None

        return predictions, probabilities
    
