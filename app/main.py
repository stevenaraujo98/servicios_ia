from typing import Union, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult

from .projects.ods import predict_ods_text
from .projects.patente import predict_patent_text
from .projects.carrera import predict_carrera_text
from .projects.objetivo import calificate_objective, calificate_objectives_gen_esp

# --- Importaciones de tu proyecto ---
from .validations import validate_min_length, validate_not_empty, clean_text
from .modelsEntity import ItemContent, ItemContentObjectives, ItemModelContent, ItemModelContentObjectives, PredictionResponse, PredictionResponseODS, PredictionResponseCareer, PredictionResponseClassificationObjective, FullEvaluationResponse, TaskCreationResponse, TaskStatusResponse
from .models.ModelLoader import ModelLoader

# --- Importaciones de Celery ---
# Importamos la instancia de Celery y la tarea específica
from .tasks import celery_app, run_objective_evaluation_task

app = FastAPI()

origins = [
    "*",
    # "https://integradora.espol.edu.ec",
]

# Configuración de CORS 
app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
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
    return {"Hello": "IA"}



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



# Calificador Objetivo
# @app.get("/predict/objetivo/")
# def read_objetivo():
#     objetivo = "Aumentar la satisfacción del cliente en un 15% para el tercer trimestre de 2025, implementando un nuevo sistema de soporte en línea y capacitando al equipo de atención al cliente."
#     print(f"Objective text: {objetivo[:100]}...")

#     model_name = "gemma3"
#     approved, verbs, detail, suggestions, suggestion_options = calificate_objective(model_name, objetivo)

#     print(f"Approved: {approved}")
#     print(f"Verbs: {verbs}")
#     print(f"Detail: {detail}")
#     print(f"Suggestions: {suggestions}")
#     print(f"Suggestion Options: {suggestion_options}")

#     return {"model_name": model_name, "approved": approved, "verbs": verbs, "detail": detail, "suggestions": suggestions, "suggestion_options": suggestion_options}

@app.post("/predict/objetivo/", response_model=PredictionResponseClassificationObjective)
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

@app.post("/predict/objetivo/{model_name}", response_model=PredictionResponseClassificationObjective)
def predict_objetivo(model_name: str, item: ItemContent, q: Union[str, None] = None):
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



# # Calificador Objetivos
# @app.get("/predict/objetivos/")
# def read_objetivos():
#     objetivo = "Desarrollando un diseño que permita la visualización de la curva I-V de un panel PV mediante la implementación de un método práctico, programable, para que pueda ser replicado por estudiantes de pregrado."

#     objetivos_especificos = [
#         "Desarrollar un procedimiento para la determinación de la curva de operación I-V para la obtención de mediciones de forma automática.",
#         "Diseñar un prototipo escalable basado en el método seleccionado para el trazador de curvas I-V",
#         "Realizar las mediciones de corriente y voltaje de un panel PV que nos permitan la adquisición diferentes puntos de la curva."
#     ]

#     model_name = "gemma3"
#     alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(model_name, objetivo, objetivos_especificos)

#     print(f"Approved: {alineacion_aprobada}")
#     print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
#     print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
#     print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")

#     response_data = {
#         "joint_evaluation": evaluacion_conjunta,
#         "individual_evaluation": evaluacion_individual
#     }    
#     return response_data

@app.post("/predict/objetivos/", response_model=FullEvaluationResponse)
def predict_objetivos(item: ItemModelContentObjectives, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")

    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    objetivos_especificos = item.specific_objectives
    if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
        raise ValueError("La lista de objetivos específicos no puede estar vacía, tampoco menos de 3 ni más de 4.")

    alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(model_name, objetivo, objetivos_especificos)

    print(f"Approved: {alineacion_aprobada}")
    print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
    print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
    print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")

    response_data = {
        "joint_evaluation": evaluacion_conjunta,
        "individual_evaluation": evaluacion_individual
    }
    
    return response_data

@app.post("/predict/objetivos/{model_name}", response_model=FullEvaluationResponse)
def predict_objetivos(model_name: str, item: ItemContentObjectives, q: Union[str, None] = None):
    if q:
        print(f"Query parameter q: {q}")
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    # validate objetivo min limit_min
    validate_min_length(objetivo, min_length=10)

    objetivos_especificos = item.specific_objectives
    if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
        raise ValueError("La lista de objetivos específicos no puede estar vacía, tampoco menos de 3 ni más de 4.")

    alineacion_aprobada, evaluacion_conjunta, evaluacion_individual = calificate_objectives_gen_esp(model_name, objetivo, objetivos_especificos)

    print(f"Approved: {alineacion_aprobada}")
    print(f"Detail: {evaluacion_conjunta['alignment_detail']}")
    print(f"Suggestion: {evaluacion_conjunta['global_suggestion']}")
    print(f"Verbs del objetivo general: {evaluacion_individual['general_objective']['verbs']}")
    
    response_data = {
        "joint_evaluation": evaluacion_conjunta,
        "individual_evaluation": evaluacion_individual
    }
    
    return response_data

# --- ENDPOINT MODIFICADO ---
# Este endpoint ahora INICIA la tarea y responde inmediatamente.
@app.post("/predict/objetivos_async/", response_model=TaskCreationResponse, status_code=202)
def predict_objetivos_async(item: ItemModelContentObjectives):
    # 1. Validaciones (se mantienen igual)
    model_name = item.model_name.strip()
    validate_not_empty(model_name)

    objetivo = clean_text(item.content)
    validate_min_length(objetivo, min_length=10)

    objetivos_especificos = item.specific_objectives
    if not objetivos_especificos or len(objetivos_especificos) < 3 or len(objetivos_especificos) > 4:
        raise HTTPException(status_code=400, detail="La lista de objetivos específicos debe contener entre 3 y 4 elementos.")

    # 2. Iniciar la tarea en segundo plano
    # En lugar de llamar a la función directamente, usamos .delay()
    # Esto envía la tarea a la cola de RabbitMQ y no espera.
    task = run_objective_evaluation_task.delay(
        model_name, objetivo, objetivos_especificos
    )

    # 3. Responder inmediatamente con el ID de la tarea
    return {"task_id": task.id, "status": "Processing"}

# --- NUEVO ENDPOINT: Consultar estado de la tarea ---
@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Consulta el estado de una tarea de fondo."""
    task_result = AsyncResult(task_id, app=celery_app)
    
    response_data = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.successful():
        response_data["result"] = task_result.get()
    elif task_result.failed():
        # Si falló, puedes optar por devolver el error
        response_data["result"] = str(task_result.info) # 'info' contiene la excepción

    return response_data

# --- NUEVO ENDPOINT: Obtener resultado de la tarea (alternativa) ---
@app.get("/tasks/{task_id}/result", response_model=FullEvaluationResponse)
def get_task_result(task_id: str):
    """Obtiene el resultado de una tarea completada."""
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready():
        raise HTTPException(status_code=202, detail="La tarea aún no ha finalizado.")
    
    if task_result.failed():
        raise HTTPException(status_code=500, detail=f"La tarea falló: {task_result.info}")

    return task_result.get()