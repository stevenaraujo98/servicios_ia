from pydantic import BaseModel

class ItemContent(BaseModel):
    content: str | None = None

class ItemModelContent(ItemContent):
    model_name: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    predictions: list[int]
    probabilities: list[list[float]] | None = None

class PredictionResponseODS(BaseModel):
    prediction: int
    probability: float | None = None
    predictions: list[list[int]] | None = None
    probabilities: list[list[float]] | None = None

class PredictionResponseCareer(BaseModel):
    prediction: str
    probability: float | None = None
    predictions: list[list[str]] | None = None
    probabilities: list[list[float]] | None = None