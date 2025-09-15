from app.consts import models_openRouter
from dotenv import load_dotenv
from ollama import chat
import requests
import json
import os

def request_ollama_chat(model, prompt="hola"):
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

def request_open_router_chat(model, prompt="What is the meaning of life?"):
    load_dotenv()

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY"),
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("HTTP_REFERER"), # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": os.getenv("X_TITLE"), # Optional. Site title for rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
    )

    json_response = response.json()
    print(json_response["id"], json_response["provider"], json_response["model"], json_response["created"])
    print(json_response["usage"])

    return json_response["choices"][0]["message"]["content"]

def get_prompt_objetivos_gen_esp(objetivo, objetivos_especificos):
    objetivos_especificos = "; ".join([f'\"{obj}\" ' for obj in objetivos_especificos])

    return '### ROL Y OBJETIVO ###\
        Eres un evaluador experto en metodología de la investigación, especializado en la coherencia y formulación de objetivos para tesis de pregrado. Tu tarea es analizar un objetivo general y sus correspondientes objetivos específicos, determinar si están correctamente formulados y si son coherentes entre sí. Finalmente, debes devolver tu evaluación en un formato JSON estricto.\
        \
        ### ESTRUCTURA DE ENTRADA ###\
        Recibirás un objeto JSON con dos claves:\
        - `objetivo_general`: Un string con el objetivo principal de la tesis.\
        - `objetivos_especificos`: Una lista de strings, cada uno siendo un objetivo específico.\
        \
        ### CONTEXTO: TAXONOMÍA DE BLOOM ###\
        El verbo principal de cada objetivo debe pertenecer a uno de los siguientes niveles cognitivos, que generalmente siguen una jerarquía:\
        - **CONOCIMIENTO/RECORDAR**: definir, listar, nombrar, identificar.\
        - **COMPRENSIÓN**: interpretar, resumir, clasificar, explicar, describir.\
        - **APLICACIÓN**: aplicar, usar, implementar, demostrar.\
        - **ANÁLISIS**: analizar, comparar, categorizar, diagnosticar, diferenciar.\
        - **SÍNTESIS/CREAR**: crear, diseñar, planificar, proponer, formular.\
        - **EVALUACIÓN**: evaluar, juzgar, criticar, valorar, justificar.\
        \
        ### PROCESO DE RAZONAMIENTO OBLIGATORIO ###\
        Para cada objetivo, sigue estos pasos en orden estricto:\
        1.  **Verificación de Verbo Inicial**: Examina la PRIMERA palabra del objetivo. ¿Es un verbo en infinitivo que termina en -ar, -er, o -ir?\
        2.  **Decisión Inmediata**: Si la respuesta al paso 1 es NO, marca `aprobado` como "NO" inmediatamente. En el campo `detalle`, cita la regla B1 y especifica que el objetivo no comienza con un infinitivo. No continúes con la evaluación S.M.A.R.T. para este objetivo.\
        3.  **Evaluación Completa**: Solo si la respuesta al paso 1 es SÍ, procede a evaluar los demás criterios (B2, B3).\
        \
        ### CRITERIOS DE EVALUACIÓN ###\
        \
        #### A. Criterios de Evaluación Global (Coherencia del Conjunto)\
        1.  **Alineación Lógica**: Los objetivos específicos DEBEN ser los pasos lógicos y necesarios que, en su conjunto, permiten alcanzar el objetivo general. Deben ser un desglose directo del objetivo principal.\
        2.  **Jerarquía Cognitiva**: Por lo general, los verbos de los objetivos específicos deben ser de un nivel cognitivo igual o inferior al del objetivo general.\
        3.  **Sintaxis de verbos**: Todos los verbos principales o iniciales de deben estar en infinitivo (ar, er o ir).\
        \
        #### B. Criterios de Evaluación Individual (Para CADA Objetivo)\
        1.  **Verbo Inicial (REGLA CRÍTICA E INNEGOCIABLE)**: Cada objetivo DEBE comenzar con un único verbo en infinitivo (terminado en -ar, -er, -ir). **No hay excepciones**. Si esta regla no se cumple, el objetivo se reprueba automáticamente (`aprobado`: "NO"), sin importar qué tan bien esté formulado el resto del texto.\
        2.  **Verbos Secundarios**: Si existen otros verbos en la oración, no deben estar en infinitivo. Deben complementar la acción principal.\
        3.  **Estructura S.M.A.R.T. Simplificada**: Cada objetivo debe responder claramente a tres preguntas:\
            * **¿Qué se hará?** (La acción principal, definida por el verbo).\
            * **¿Cómo se hará?** (El método, las herramientas o el proceso).\
            * **¿Para qué se hará?** (El propósito, la finalidad o el impacto esperado).\
        \
        \
        ### FORMATO DE SALIDA Y REGLAS ###\
        Tu respuesta DEBE ser exclusivamente un objeto JSON válido y nada más. No incluyas texto introductorio ni explicaciones fuera del JSON, tampoco menciones textualmente las letras A y B como items a cumplir. La estructura será la siguiente:\
        \
        - **`evaluacion_conjunta`**: Objeto con la evaluación de la alineación entre objetivos.\
            - **`alineacion_aprobada`**: "SI" o "NO". Marca "NO" si falla el criterio A1 o A2.\
            - **`detalle_alineacion`**: Explicación clara de por qué el conjunto es coherente o no.\
            - **`sugerencia_global`**: Una recomendación general para mejorar la relación entre los objetivos.\
        - **`evaluacion_individual`**: Objeto que contiene la evaluación de cada objetivo por separado.\
            - **`objetivo_general`**: Un objeto con la evaluación individual del objetivo general.\
                - `aprobado`: "SI" o "NO", según los criterios B1, B2, B3.\
                - `verbos`: Una lista de strings. Incluye cualquier verbo en infinitivo que esté mal utilizado o que no cumpla con la estructura de inicio en infinitivo. Si no hay errores de verbos, deja la lista vacía `[]`.\
                - `detalle`: Explicación de la aprobación o rechazo individual.\
                - `sugerencias`: Cómo mejorar este objetivo específico.\
                - `opciones_de_sugerencias`: Si `aprobado` es "NO", proporciona 2 reescrituras corregidas. Si es "SI", deja la lista vacía `[]`.\
            - **`objetivos_especificos`**: Una lista de objetos, donde cada objeto evalúa un objetivo específico.\
                - `objetivo`: El texto del objetivo específico evaluado.\
                - `aprobado`: "SI" o "NO", según los criterios B1, B2, B3.\
                - `detalle`: Explicación de la aprobación o rechazo individual.\
                - `sugerencias`: Cómo mejorar este objetivo específico.\
                - `opciones_de_sugerencias`: Si `aprobado` es "NO", proporciona 2 reescrituras corregidas. Si es "SI", deja la lista vacía `[]`.\
        \
        ### EJEMPLOS (FEW-SHOT LEARNING) ###\
        \
        **Ejemplo 1: Objetivos BIEN formulados y alineados**\
        "input": {\
        "objetivo_general": "Diseñar un plan de comunicación digital para la marca \'EcoVida\' utilizando análisis de redes sociales para incrementar su posicionamiento en el mercado local durante el próximo semestre.",\
        "objetivos_especificos": [\
            "Diagnosticar la situación actual de la comunicación digital de \'EcoVida\' mediante un análisis FODA.",\
            "Identificar el público objetivo principal de la marca a través de encuestas y análisis de datos demográficos de sus seguidores actuales.",\
            "Proponer tres estrategias de contenido específicas para Instagram y TikTok basadas en las mejores prácticas del sector."\
        ]\
        }\
        "respuesta":\
        {\
        "evaluacion_conjunta": {\
            "alineacion_aprobada": "SI",\
            "detalle_alineacion": "Los objetivos específicos son pasos lógicos y secuenciales (diagnosticar, identificar, proponer) que conducen directamente al diseño del plan general. La jerarquía cognitiva es correcta: los verbos de los específicos (Diagnosticar - Análisis, Identificar - Conocimiento, Proponer - Síntesis) son de nivel igual o inferior a Diseñar (Síntesis).",\
            "sugerencia_global": "La estructura es sólida. Para fortalecerla aún más, se podría añadir un objetivo específico relacionado con la medición o KPIs del plan."\
        },\
        "evaluacion_individual": {\
            "objetivo_general": {\
            "aprobado": "SI",\
            "verbos": [],\
            "detalle": "Correctamente formulado. Responde al qué, cómo y para qué.",\
            "sugerencias": "El objetivo es claro y completo.",\
            "opciones_de_sugerencias": []\
            },\
            "objetivos_especificos": [\
            {\
                "objetivo": "Diagnosticar la situación actual de la comunicación digital de \'EcoVida\' mediante un análisis FODA.",\
                "aprobado": "SI",\
                "detalle": "Correcto. Verbo inicial de análisis, define el qué y el cómo.",\
                "sugerencias": "Para mayor claridad, se podría añadir el \'para qué\', por ejemplo: \'...para establecer la línea base del plan\'.",\
                "opciones_de_sugerencias": []\
            },\
            {\
                "objetivo": "Identificar el público objetivo principal de la marca a través de encuestas y análisis de datos demográficos de sus seguidores actuales.",\
                "aprobado": "SI",\
                "detalle": "Correcto. Verbo inicial de conocimiento, especifica el qué y el cómo.",\
                "sugerencias": "El objetivo está bien definido.",\
                "opciones_de_sugerencias": []\
            },\
            {\
                "objetivo": "Proponer tres estrategias de contenido específicas para Instagram y TikTok basadas en las mejores prácticas del sector.",\
                "aprobado": "SI",\
                "detalle": "Correcto. Verbo de síntesis que se alinea con el objetivo general. Define qué y cómo.",\
                "sugerencias": "El objetivo es concreto y medible.",\
                "opciones_de_sugerencias": []\
            }\
            ]\
        }\
        }\
        \
        \
        **Ejemplo 2: Objetivos MAL formulados y no alineados**\
        "input": {\
        "objetivo_general": "Analizar el impacto de las políticas de trabajo remoto en la productividad de los empleados de la empresa XYZ.",\
        "objetivos_especificos": [\
            "Se describirán las políticas actuales de la empresa.",\
            "Crear una nueva plataforma de software para gestionar el trabajo remoto.",\
            "Evaluar la satisfacción de los empleados."\
        ]\
        }\
        "respuesta":\
        {\
        "evaluacion_conjunta": {\
            "alineacion_aprobada": "NO",\
            "detalle_alineacion": "El conjunto falla en la alineación. El específico 2 (\'Crear una nueva plataforma\') es de un nivel cognitivo (Crear) superior al general (\'Analizar\') y excede el alcance de un simple análisis. El específico 3 (\'Evaluar\') es pertinente, pero el 2 rompe la coherencia lógica.",\
            "sugerencia_global": "Los objetivos específicos deben ser un desglose del análisis propuesto, no proponer la creación de soluciones. Deberían enfocarse en qué aspectos de la productividad se analizarán y cómo, por ejemplo: comparar métricas, interpretar encuestas, etc."\
        },\
        "evaluacion_individual": {\
            "objetivo_general": {\
            "aprobado": "SI",\
            "verbos": [],\
            "detalle": "El objetivo general está bien formulado, aunque podría especificar el \'cómo\'.",\
            "sugerencias": "Para mejorar, especificar la metodología: \'...a través de encuestas de percepción y análisis de métricas de rendimiento\'."\
            "opciones_de_sugerencias": []\
            },\
            "objetivos_especificos": [\
            {\
                "objetivo": "Se describirán las políticas actuales de la empresa.",\
                "aprobado": "NO",\
                "detalle": "Falla el criterio B1. No comienza con un verbo en infinitivo. La redacción es pasiva.",\
                "sugerencias": "Debe iniciar con un verbo de acción en infinitivo.",\
                "opciones_de_sugerencias": [\
                "Describir las políticas de trabajo remoto implementadas por la empresa XYZ desde 2020.",\
                "Listar los principales componentes de la normativa de teletrabajo vigente en la empresa XYZ."\
                ]\
            },\
            {\
                "objetivo": "Crear una nueva plataforma de software para gestionar el trabajo remoto.",\
                "aprobado": "SI",\
                "detalle": "Individualmente, el objetivo está bien formulado (qué y para qué implícito). Sin embargo, no está alineado con el objetivo general.",\
                "sugerencias": "Este objetivo es demasiado ambicioso para una tesis cuyo alcance es solo \'analizar\'. Debería ser un proyecto de desarrollo, no un objetivo específico de una investigación analítica.",\
                "opciones_de_sugerencias": []\
            },\
            {\
                "objetivo": "Evaluar la satisfacción de los empleados.",\
                "aprobado": "NO",\
                "detalle": "Falla el criterio B3. Es demasiado vago. No especifica el \'cómo\' ni el \'para qué\' en relación a la productividad.",\
                "sugerencias": "Debe conectarse explícitamente con la productividad y detallar el método.",\
                "opciones_de_sugerencias": [\
                "Medir el nivel de satisfacción de los empleados con el trabajo remoto mediante una encuesta estandarizada para correlacionarlo con su rendimiento.",\
                "Valorar la percepción de los empleados sobre cómo las políticas de trabajo remoto afectan su bienestar y productividad, utilizando entrevistas semiestructuradas."\
                ]\
            }\
            ]\
        }\
        }\
        \
        \
        **Ejemplo 3: Objetivo que falla por usar gerundio**\
        "input": {\
        "objetivo_general": "Desarrollando un sistema de monitoreo para optimizar el consumo de energía.",\
        "objetivos_especificos": [\
            "Identificar los puntos de mayor consumo."\
        ]\
        }\
        "respuesta":\
        {\
        "evaluacion_conjunta": {\
            "alineacion_aprobada": "NO",\
            "detalle_alineacion": "El objetivo general no cumple con el formato requerido, lo que invalida la evaluación conjunta.",\
            "sugerencia_global": "Corregir la formulación del objetivo general para que comience con un verbo en infinitivo."\
        },\
        "evaluacion_individual": {\
            "objetivo_general": {\
            "aprobado": "NO",\
            "verbos": ["Desarrollando"],\
            "detalle": "Falla la regla crítica B1. El objetivo comienza con el gerundio \'Desarrollando\' en lugar de un verbo en infinitivo como \'Desarrollar\'.",\
            "sugerencias": "El objetivo siempre debe iniciar con un verbo de acción en infinitivo.",\
            "opciones_de_sugerencias": [\
                "Desarrollar un sistema de monitoreo para optimizar el consumo de energía.",\
                "Crear un sistema de monitoreo que permita optimizar el consumo de energía."\
            ]\
            },\
            "objetivos_especificos": [\
            {\
                "objetivo": "Identificar los puntos de mayor consumo.",\
                "aprobado": "SI",\
                "detalle": "Correcto. Verbo inicial de conocimiento, aunque le falta especificar el cómo y el para qué para ser más robusto.",\
                "sugerencias": "El objetivo es claro pero podría ser más completo.",\
                "opciones_de_sugerencias": []\
            }\
            ]\
        }\
        }\
        \
        ### OBJETIVOS A EVALUAR ###\
        """{"objetivo_general": "' + objetivo + '",\
        "objetivos_especificos": [' + objetivos_especificos + ']}"""'

def extract_json_from_response(only_json):
    """ Extrae y estructura el JSON de la respuesta del modelo. """
    json_dict = json.loads(only_json)

    joint_evaluation = {}
    joint_evaluation["alignment_approved"] = True if json_dict["evaluacion_conjunta"]["alineacion_aprobada"] == "SI" else False
    joint_evaluation["alignment_detail"] = json_dict["evaluacion_conjunta"]["detalle_alineacion"]
    joint_evaluation["global_suggestion"] = json_dict["evaluacion_conjunta"]["sugerencia_global"]

    individual_evaluation = {}
    individual_evaluation["general_objective"] = {}
    individual_evaluation["specific_objectives"] = []

    individual_evaluation["general_objective"]["approved"] = True if json_dict["evaluacion_individual"]["objetivo_general"]["aprobado"] == "SI" else False
    individual_evaluation["general_objective"]["verbs"] = json_dict["evaluacion_individual"]["objetivo_general"]["verbos"]
    individual_evaluation["general_objective"]["detail"] = json_dict["evaluacion_individual"]["objetivo_general"]["detalle"]
    individual_evaluation["general_objective"]["suggestions"] = json_dict["evaluacion_individual"]["objetivo_general"]["sugerencias"]
    individual_evaluation["general_objective"]["suggestion_options"] = json_dict["evaluacion_individual"]["objetivo_general"]["opciones_de_sugerencias"]

    for i in json_dict["evaluacion_individual"]["objetivos_especificos"]:
        individual_evaluation["specific_objectives"].append({})
        individual_evaluation["specific_objectives"][-1]["objective"] = i["objetivo"]
        individual_evaluation["specific_objectives"][-1]["approved"] = True if i["aprobado"] == "SI" else False
        individual_evaluation["specific_objectives"][-1]["detail"] = i["detalle"]
        individual_evaluation["specific_objectives"][-1]["suggestions"] = i["sugerencias"]
        individual_evaluation["specific_objectives"][-1]["suggestion_options"] = i["opciones_de_sugerencias"]
    
    return joint_evaluation, individual_evaluation

def calificate_objectives_gen_esp(model_name, general, especificos):
    """ Versión usando ollama """
    print("Solicitud con modelo:", model_name)
    prompt_target = get_prompt_objetivos_gen_esp(general, especificos)
    print("Prompt generado ...", prompt_target[-400:])

    response = request_ollama_chat(model=model_name, prompt=prompt_target)
    print("Respondio el modelo")
    only_json = response[response.find("{"):response.rfind("}") + 1]
    print("JSON extraido:", only_json)

    joint_evaluation, individual_evaluation = extract_json_from_response(only_json)

    return joint_evaluation["alignment_approved"], joint_evaluation, individual_evaluation

def calificate_objectives_gen_esp_simple(model_name, general, especificos):
    """ Versión simplificada usando OpenRouter.ai """
    if model_name not in models_openRouter.keys():
        raise ValueError(f"Modelo '{model_name}' no está disponible en OpenRouter.ai. Modelos disponibles: {list(models_openRouter.keys())}")
    print("Solicitud con modelo:", model_name)
    
    prompt_target = get_prompt_objetivos_gen_esp(general, especificos)
    print("Prompt generado ...", prompt_target[-400:])

    content_response = request_open_router_chat(model=models_openRouter[model_name], prompt=prompt_target)
    
    print("Respondio el modelo")
    only_json = content_response[content_response.find("{"):content_response.rfind("}") + 1]
    print("JSON extraido:", only_json)

    joint_evaluation, individual_evaluation = extract_json_from_response(only_json)

    return joint_evaluation["alignment_approved"], joint_evaluation, individual_evaluation
