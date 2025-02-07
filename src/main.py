from typing import Optional
import os
import typer
from backend.run_backend import BackendRunner
from benchmark.test_backend_working import test_backend_working
from model.get_models import get_models
from loguru import logger
from hardware.hardware_cli import display_hardware_menu
from hardware.hardware_detector import HardwareDetector
from model.weights import download_no_weights_model
from huggingface_hub.utils._auth import get_token
from backend.backend_types import get_backend_types
from dataset.leaderboard_dataset import LeaderboardData, upload_data_to_hub
from datetime import datetime
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()

if get_token() is None:
    raise ValueError(
        "To run the benchmarks you need to set the HuggingFace token, please run huggingface-cli login"
    )

app = typer.Typer()

machine = os.environ.get("MACHINE_NAME", "unknown")
if machine == "unknown":
    logger.debug("(CI) Machine is not set so the results will not be uploaded to the leaderboard")

@app.command()
def start_benchmark(
    no_weights: Optional[bool] = False,
):
    """
    Start all the benchmarks for all the top text generation models on Hugging Face Hub with all the backends.
    Args:
        no_weights (Optional[bool], optional): Whether to download models without weights. Defaults to True.
    """
    # Check if hardware type is already set in environment
    selected_hardware = os.environ.get("HARDWARE_TYPE")
    
    if selected_hardware is None:        
        # Display menu and get selection
        detector = HardwareDetector()
        # Detect available hardware
        recommended_hardware = detector.get_recommended_hardware()
        # Display menu and get selection
        selected_hardware = display_hardware_menu(recommended_hardware)

    console.print(f"\n[green]Selected Hardware Configuration:[/green]: {selected_hardware}\n")

    top_models = get_models()
    logger.info(
        f"\nTop {len(top_models)} text generation models on Hugging Face Hub:"
    )
    
    results: list[LeaderboardData] = []
    for i, model in enumerate(top_models, 1):
        logger.info(f"{i}. {model.name}")
        for backend_type in get_backend_types():
            logger.info(
                f"Running benchmark for {model.name} with {backend_type} backend"
            )

            result = single_model_benchmark(model.hf_model_id, backend_type, selected_hardware, no_weights)
            results.append(result)
            if result.can_serve_single_request and os.environ.get("QUICK_BENCHMARKING", "0") == "1":
                logger.info(f"Quick benchmarking enabled, stopping after first successful benchmark")
                break
                
                
    # display summary of results
    console.print("\n[green]Summary of results:[/green]")
    for result in results:
        console.print(f"{selected_hardware} - {result.model_id} with {result.backend_type} backend: {'[green]working[/green]' if result.can_serve_single_request else '[red]failed[/red]'}")
    
    # Upload all results at once
    upload_data_to_hub(results)


def single_model_benchmark(
    model_id: str,
    backend_type: str,
    hardware_type: str,
    no_weights: Optional[bool] = True,
) -> LeaderboardData:
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
        raise ValueError("No weights are currently not working")
        logger.info(f"Using no weights model {model_id}")
        download_no_weights_model(model_id)

    backend_runner = BackendRunner()
    
    can_serve_single_request = False
    
    try:
        # Start the backend server
        backend_runner.run(model_id, backend_type, hardware_type, no_weights)
        
        # Try the requests - this will try chat first, then completion if chat fails
        if test_backend_working(model_id):
            can_serve_single_request = True
    finally:
        # Only stop the backend after all attempts are done
        backend_runner.stop()
        
    return LeaderboardData(
        model_id=model_id,
        backend_type=backend_type,
        can_serve_single_request=can_serve_single_request,
        hardware_type=hardware_type,
        machine=machine,
        benchmark_time=datetime.now()
    )

if __name__ == "__main__":
    app()