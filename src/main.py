from typing import Optional
import os
import typer
from backend.run_backend import BackendRunner
from benchmark.single_request import try_single_request
from get_top_models import (
    get_top_text_generation_models,
    NUMBER_OF_MODELS_TO_BENCHMARK,
)
from loguru import logger
from weights import download_no_weights_model
from huggingface_hub.utils._auth import get_token
from backend.backend_types import get_backend_types
from results import LeaderboardData, upload_data_to_hub

if get_token() is None:
    raise ValueError(
        "To run the benchmarks you need to set the HuggingFace token, please run huggingface-cli login"
    )

app = typer.Typer()

machine = os.environ.get("MACHINE", "unknown")

@app.command()
def start_benchmarks(
    no_weights: Optional[bool] = True,
):
    """
    Start all the benchmarks for all the top text generation models on Hugging Face Hub with all the backends.
    Args:
        no_weights (Optional[bool], optional): _description_. Defaults to True.
    """
    top_models = get_top_text_generation_models()
    logger.info(
        f"\nTop {NUMBER_OF_MODELS_TO_BENCHMARK} text generation models on Hugging Face Hub:"
    )
    
    results = []
    for i, model in enumerate(top_models, 1):
        logger.info(f"{i}. {model['model_id']}: {model['downloads']:,} downloads")
        for backend_type in get_backend_types():
            logger.info(
                f"Running benchmark for {model['model_id']} with {backend_type} backend"
            )
            try:
                start_benchmark(model["model_id"], backend_type, no_weights)
                result = LeaderboardData(
                    model_id=model["model_id"],
                    backend_type=backend_type,
                    working=True,
                    machine=machine
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run benchmark for {model['model_id']} with {backend_type} backend: {str(e)}")
                # Add failed result to track failures
                results.append(
                    LeaderboardData(
                        model_id=model["model_id"],
                        backend_type=backend_type,
                        working=False,
                        machine=machine
                    )
                )
    
    # Upload all results at once
    try:
        upload_data_to_hub(results)
        logger.info("Successfully uploaded results to the leaderboard")
    except Exception as e:
        logger.error(f"Failed to upload results to the leaderboard: {str(e)}")


@app.command()
def start_benchmark(
    model_id: str,
    backend_type: str,
    no_weights: Optional[bool] = True,
):
    """
    Start a benchmark for a given model and backend.
    Args:
        model_id (str): The model identifier to use such as "openai-community/gpt2"
        type (BackendType): The backend type to use such as "vllm" or "tgi"
        no_weights (bool, optional): Whether to download the model without weights. Defaults to True.
    """
    if backend_type not in get_backend_types():
        raise ValueError(f"backend_type must be one of the following: {get_backend_types()}")

    if no_weights:
        download_no_weights_model(model_id)

    backend_runner = BackendRunner()
    
    try:
        backend_runner.run(model_id, backend_type, no_weights)
        try_single_request()
    finally:
        backend_runner.stop()


if __name__ == "__main__":
    app()
