import os
import shutil
import torch
from transformers import PretrainedConfig
from safetensors.torch import save_file
from loguru import logger
from huggingface_hub.constants import HF_HOME
from huggingface_hub.file_download import repo_folder_name

NO_WEIGHTS_CACHE_DIR = os.path.join(HF_HOME, "no_weights_models")

if os.getenv("CLEAN_CACHE_DIR", "0") == "1" and os.path.exists(NO_WEIGHTS_CACHE_DIR):
    shutil.rmtree(NO_WEIGHTS_CACHE_DIR)


def download_no_weights_model(model_id: str):
    # Create base no_weights directory
    os.makedirs(NO_WEIGHTS_CACHE_DIR, exist_ok=True)

    model_folder_name = repo_folder_name(repo_id=model_id, repo_type="model")
    model_folder_path = os.path.join(HF_HOME, model_folder_name)
    no_weights_model_path = os.path.join(NO_WEIGHTS_CACHE_DIR, model_folder_name)

    # if the model is already in the no_weights_models set or if the model is already downloaded, do nothing
    if os.path.exists(no_weights_model_path):
        return

    if os.path.exists(model_folder_path):
        os.symlink(model_folder_path, no_weights_model_path)
        return

    try:
        pretrained_config = PretrainedConfig.from_pretrained(model_id)
    except Exception as e:
        raise ValueError(f"Failed to load config from {model_id}: {str(e)}")

    # Create and save dummy state dict, this is our dummy state to replace the weights download
    state_dict = torch.nn.Linear(1, 1).state_dict()

    # Create the model directory before saving files
    os.makedirs(no_weights_model_path, exist_ok=True)

    # Save safetensors file
    safetensors_path = os.path.join(no_weights_model_path, "model.safetensors")
    save_file(tensors=state_dict, filename=safetensors_path, metadata={"format": "pt"})

    # Save config
    logger.info("Saving model config")
    pretrained_config.save_pretrained(save_directory=no_weights_model_path)
