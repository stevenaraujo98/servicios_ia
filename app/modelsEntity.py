from pydantic import BaseModel

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
    Clase que extiende ItemContent para incluir el nombre del modelo
    Esta clase se utiliza para enviar el nombre del modelo junto con el contenido
"""
class ItemModelContent(ItemContent):
    model_name: str

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