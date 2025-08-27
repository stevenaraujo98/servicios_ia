from typing import Union
from validations import validate_min_length, validate_not_empty, clean_text
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projects.ods import predict_ods_text
from projects.patente import predict_patent_text
from projects.carrera import predict_carrera_text
from projects.objetivo import calificate_objective
from models.ModelLoader import ModelLoader
from modelsEntity import ItemContent, ItemModelContent, PredictionResponse, PredictionResponseODS, PredictionResponseCareer, PredictionResponseClasificationObjective

app = FastAPI()

origin = "*"

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cargador de modelos
loader_ods = ModelLoader()
loader_patente = ModelLoader(tipo='patente')
loader_carrera = ModelLoader(tipo='carrera')

print("Cargando modelos de transformadores...")
loader_ods.load_transformer_model("distilbert_10e_24b_0") # Cashear el modelo para evitar recargas innecesarias
# loader_patente.load_transformer_model("distilbert_20250710_0233") # Cashear el modelo para evitar recargas innecesarias
# 1.2.2 de kaggle
# loader_carrera.load_transformer_model("bert_20250806_234119") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos de transformadores...")

print("Cargando modelos tradicionales...")
# loader_ods.load_traditional_model("Logistic_Regression_20250611_165546") # Cashear el modelo para evitar recargas innecesarias
loader_patente.load_traditional_model("Random_Forest_20250813_144340") # Cashear el modelo para evitar recargas innecesarias
loader_carrera.load_traditional_model("Random_Forest_20250808_161322") # Cashear el modelo para evitar recargas innecesarias
print("Finalizó carga de modelos tradicionales...")

@app.get("/")
def read_root():
    return {"Hello": "World"}



# PATENTE
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

@app.post("/predict/patente/", response_model=PredictionResponse)
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

@app.post("/predict/patente/{model_name}", response_model=PredictionResponse)
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

@app.post("/predict/ods/", response_model=PredictionResponseODS)
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

@app.post("/predict/ods/{model_name}", response_model=PredictionResponseODS)
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



# CARRERA
# @app.get("/predict/carrera/")
# def read_root():
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

@app.post("/predict/carrera/", response_model=PredictionResponseCareer)
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

@app.post("/predict/carrera/{model_name}", response_model=PredictionResponseCareer)
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



# gemma3:12b, 4b, deepseek-r1:14b, 8b
# Calificador Objetivo
@app.get("/predict/objetivo/")
def read_root():
    objetivo = "Aumentar la satisfacción del cliente en un 15% para el tercer trimestre de 2025, implementando un nuevo sistema de soporte en línea y capacitando al equipo de atención al cliente."
    print(f"Objective text: {objetivo[:100]}...")

    model_name = "gemma3"
    approved, verbs, detail, suggestions, suggestion_options = calificate_objective(model_name, objetivo)

    print(f"Approved: {approved}")
    print(f"Verbs: {verbs}")
    print(f"Detail: {detail}")
    print(f"Suggestions: {suggestions}")
    print(f"Suggestion Options: {suggestion_options}")

    return {"model_name": model_name, "approved": approved, "verbs": verbs, "detail": detail, "suggestions": suggestions, "suggestion_options": suggestion_options}

@app.post("/predict/objetivo/", response_model=PredictionResponseClasificationObjective)
def predict_objetivo(item: ItemModelContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    approved, verbs, detail, suggestions, suggestion_options = calificate_objective(model_name, objetivo)

    print(f"Approved: {approved}")
    print(f"Verbs: {verbs}")
    print(f"Detail: {detail}")
    print(f"Suggestions: {suggestions}")
    print(f"Suggestion Options: {suggestion_options}")
    return {"approved": approved, "verbs": verbs, "detail": detail, "suggestions": suggestions, "suggestion_options": suggestion_options}

@app.post("/predict/carrera/{model_name}", response_model=PredictionResponseClasificationObjective)
def predict_carrera(model_name: str, item: ItemContent, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    approved, verbs, detail, suggestions, suggestion_options = calificate_objective(model_name, objetivo)

    print(f"Approved: {approved}")
    print(f"Verbs: {verbs}")
    print(f"Detail: {detail}")
    print(f"Suggestions: {suggestions}")
    print(f"Suggestion Options: {suggestion_options}")
    return {"approved": approved, "verbs": verbs, "detail": detail, "suggestions": suggestions, "suggestion_options": suggestion_options}
