from typing import List
import yaml
from pydantic import BaseModel

class Model(BaseModel):
    name: str
    hf_model_id: str
    ollama_model_name: str

def get_models() -> List[Model]:
    # load from yaml file
    with open("src/model/models.yaml", "r") as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
        
    models = [Model(**model) for model in models["models"]]
    
    return models
