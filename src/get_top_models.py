import requests
import os
from typing import List, Dict

NUMBER_OF_MODELS_TO_BENCHMARK = 10


def get_top_text_generation_models(
    n: int = NUMBER_OF_MODELS_TO_BENCHMARK, sort: str = "downloads", direction: int = -1
) -> List[Dict]:
    base_url = "https://huggingface.co/api/models"
    params = {
        "sort": sort,
        "direction": direction,
        "limit": n,
        "filter": "text-generation",
        "full": "false",
    }

    headers = {}
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    if huggingface_token:
        headers["Authorization"] = f"Bearer {huggingface_token}"

    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()  # Raise an exception for bad responses

    models = response.json()
    return [
        {
            "model_id": model["id"],
            "organization": model["id"].split("/")[0],
            "model_name": model["id"].split("/")[-1],
            "downloads": model.get("downloads", 0),
        }
        for model in models
        if "downloads" in model
    ]
