from ollama import chat
import json
from typing import Union
from fastapi import APIRouter
from ..modelsEntity import ItemContent, ItemModelContent, PredictionResponseClassificationObjective
from ..validations import validate_min_length, validate_not_empty, clean_text

objetivo_router = APIRouter()

def request_chat(model, prompt="hola"):
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
            # {"role": "user", "content": content},
            # {"role": "user", "content": question},
        ],
        # stream=True,
    )

    return response.message.content

def get_prompt_objetivo(objetivo):
    return 'Actúa como evaluador de objetivos académicos de tesis de pregrado. Recibirás el texto de un objetivo y debes determinar si está correctamente formulado y redactado. Tu salida debe limitarse únicamente a un JSON válido, sin texto adicional fuera de él.\
        Evalúa lo siguiente:\
        1) El objetivo debe comenzar con un único verbo en infinitivo de la taxonomía de Bloom (ejemplo: “Diseñar”, “Implementar”, “Analizar”).;\
        2) Puede haber varios verbos pero solo debe estar en infinitivo el principal o el del inicio, los otros verbos en la redacción son permitidos siempre que complementen el objetivo.\
        3) El objetivo debe responder de forma explícita a estas tres preguntas: ¿Qué? la acción principal del proyecto. ¿Cómo? el método, estrategia o acciones para lograrlo. ¿Para qué? el propósito o impacto esperado.;\
        4) Toma en cuenta el contexto de la carrera (como diseño, producción u otras áreas afines). En estos casos, un objetivo puede no responder explícitamente a las tres preguntas, pero no necesariamente está mal. En tal situación, debes evaluar si el nivel de claridad es suficiente y aclararlo en el campo “detalle”.;\
        5) Si alguno de los puntos falla, marca “aprobado” como “NO” y explica en el campo “detalle” qué falta o qué está incorrecto.;\
        6) En el campo "sugerencias" detalla que puede mejorar del texto del objetivo, ya sea que fue aprobado o no.\
        \
        Cuando el objetivo no cumpla, debes también devolver 3 ejemplos de objetivos alternativos bien redactados (opciones de mejora).\
        Si el objetivo cumple en todo, deja “opciones de sugerencias” como lista vacía.\
        \
        En el campo “verbos”, incluye cualquier verbo principal en infinitivo usado de manera incorrecta o mal escrito; si no hay problemas, devuelve una lista vacía. No inventes datos que no estén en el objetivo.\
        \
        El formato de salida debe ser exactamente este:\
        \
        {\
        "aprobado": "SI" | "NO",\
        "verbos": ["..."],\
        "detalle": "",\
        "sugerencias": "",\
        "opciones de sugerencias": ["", "", ""]\
        }\
        \
        Objetivo a evaluar: """' + objetivo + '"""\
        \
        Notas finales: No incluyas nada fuera del JSON. Si el objetivo es ambiguo o corresponde a un área en la que no siempre se expresan las tres preguntas, especifícalo en el campo “detalle” y sugiere en “sugerencias” cómo podría mejorar en claridad sin perder coherencia con la disciplina.'

def calificate_objective(model_name, text):
    print("Solicitud con modelo:", model_name)
    prompt_target = get_prompt_objetivo(text)
    print("Prompt generado ...", prompt_target[-100:])

    response = request_chat(model=model_name, prompt=prompt_target)
    print("Respondio el modelo")
    only_json = response[response.find("{"):response.rfind("}") + 1]
    print("JSON extraido:", only_json)

    json_dict = json.loads(only_json)
    return True if json_dict["aprobado"] == "SI" else False, json_dict["verbos"], json_dict["detalle"], json_dict["sugerencias"], json_dict["opciones de sugerencias"]

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

@objetivo_router.post("/", response_model=PredictionResponseClassificationObjective)
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

@objetivo_router.post("/{model_name}", response_model=PredictionResponseClassificationObjective)
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
