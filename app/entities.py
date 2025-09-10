from pydantic import BaseModel, Field
from typing import List, Dict, Any

"""
    Clases para objeto de entrada y salida de las aplicaciones de FastAPI
    Estas clases definen la estructura de los datos que se envían y reciben
"""

"""
    Clase base para el contenido del item
    Esta clase se utiliza para enviar el contenido del item en las peticiones
"""
class ItemContent(BaseModel):
    content: str | None = None

"""
    Clase que extiende ItemContent para incluir los objetivos específicos
"""
class ItemContentObjectives(ItemContent):
    specific_objectives: list[str]

"""
    Clase que extiende ItemContent para incluir el nombre del modelo
    Esta clase se utiliza para enviar el nombre del modelo junto con el contenido
"""
class ItemModelContent(ItemContent):
    model_name: str

"""
    Clase que extiende ItemModelContent para incluir los objetivos específicos
"""
class ItemModelContentObjectives(ItemModelContent):
    specific_objectives: list[str]

"""
    prediction: etiqueta predicha
    probability: probabilidad de la etiqueta predicha
    predictions: lista de etiquetas predichas (prediction en lista)
    probabilities: lista de probabilidades de las etiquetas predichas
"""
class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    predictions: list[int]
    probabilities: list[float] | None = None

"""
    prediction: etiqueta predicha
    probability: probabilidad de la etiqueta predicha
    predictions: lista de lista del top 3 de etiquetas predichas
    probabilities: lista de lista de todas las probabilidades de las etiquetas predichas
"""
class PredictionResponseODS(BaseModel):
    prediction: int
    probability: float | None = None
    predictions: list[list[int]] | None = None
    probabilities: list[list[float]] | None = None

"""
    prediction: etiqueta predicha
    probability: probabilidad de la etiqueta predicha
    top3_careers: lista del top 3 de etiquetas predichas
    top3_probabilities: lista de las probabilidades correspondientes al top 3 de etiquetas predichas
"""
class PredictionResponseCareer(BaseModel):
    prediction: str
    probability: float | None = None
    top3_careers: list[str] | None = None
    top3_probabilities: list[float] | None = None

"""
    approved: booleano de si o no está aprobado
    verbs: lista de los verbos mal utilizado
    detail: Una explicación clara y concisa de por qué el objetivo fue aprobado o no. Si fue rechazado, especifica qué criterios fallaron.
    suggestions: Una recomendación general sobre cómo mejorar el objetivo, incluso si fue aprobado.
    suggestion_options: lista de 3 reescrituras completas y corregidas del objetivo.
"""
class PredictionResponseClassificationObjective(BaseModel):
    approved: bool
    verbs: list[str] | None = None
    detail: str | None = None
    suggestions: str | None = None
    suggestion_options: list[str] | None = None


    

"""
    Funciones para la evaluación conjunta de objetivos
"""
class JointEvaluation(BaseModel):
    """Modelo para la sección 'evaluacion_conjunta'."""
    alignment_approved: bool
    alignment_detail: str
    global_suggestion: str

class PredictionClassificationObjective(BaseModel):
    """
    Modelo para la evaluación del objetivo general, basado en tu ejemplo.
    """
    approved: bool
    verbs: list[str] | None
    detail: str
    suggestions: list[str] | str | None
    suggestion_options: list[str] | None

class SpecificObjectiveEvaluation(BaseModel):
    """Modelo para cada uno de los objetivos específicos en la lista."""
    objective: str
    approved: bool
    detail: str
    suggestions: list[str] | str | None
    suggestion_options: list[str] | None

class IndividualEvaluation(BaseModel):
    """Modelo para la sección 'evaluacion_individual'."""
    general_objective: PredictionClassificationObjective
    specific_objectives: list[SpecificObjectiveEvaluation]

class FullEvaluationResponse(BaseModel):
    """Modelo raíz que contiene la estructura completa de la respuesta."""
    joint_evaluation: JointEvaluation# = Field(..., alias='evaluacion_conjunta') # la respuesta es con el alias
    individual_evaluation: IndividualEvaluation# = Field(..., alias='evaluacion_individual')



# --- NUEVOS MODELOS PARA TAREAS ASÍNCRONAS ---
class TaskCreationResponse(BaseModel):
    """Respuesta al iniciar una nueva tarea."""
    task_id: str
    status: str

class TaskStatusResponse(BaseModel):
    """Respuesta al consultar el estado de una tarea."""
    task_id: str
    status: str
    result: Any | None = None
