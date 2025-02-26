from jinja2 import Template
import subprocess
import time
import threading
from typing import Optional
from model.weights import NO_WEIGHTS_CACHE_DIR
from huggingface_hub.utils._auth import get_token
from hardware.hardware_info import get_hardware_info
from backend.backend_types import get_backend_types
from loguru import logger
import requests
from huggingface_hub.constants import HF_HOME
import os
from pathlib import Path

from model.get_models import Model

BENCHMARKING_CONTAINER_NAME = "llm-hardware-benchmark"


class BackendRunner:

    def __init__(self):
        self.process = None

    @staticmethod
    def _log_stream(stream):
        """Log output from a stream line by line with a prefix."""
        for line in iter(stream.readline, ''):
            if line:
                print(line.strip())

    def wait_for_server(self, backend_type, timeout=3000, check_interval=10):
        """
        Wait for the server to be ready by checking the health endpoint and container status.
        
        Args:
            timeout (int): Maximum time to wait in seconds
            check_interval (int): Time between checks in seconds
        
        Returns:
            bool: True if server is ready, False otherwise
        """   
        start_time = time.time()
        attempts = 0
        while time.time() - start_time < timeout:
            try:
                # Check if container is still running
                result = subprocess.run(
                    f"docker inspect -f '{{{{.State.Status}}}}' {BENCHMARKING_CONTAINER_NAME}",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    status = result.stdout.strip()
                    if status == 'exited':
                        # Get container logs to help with debugging
                        subprocess.run(f"docker logs {BENCHMARKING_CONTAINER_NAME}", shell=True)
                        raise RuntimeError("Container exited before becoming ready")
                    elif status != 'running':
                        raise RuntimeError(f"Container is in unexpected state: {status}")
                
                # Check if server is responding
                url = "http://localhost:8080/health"
                response = requests.get(url)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            except RuntimeError as e:
                logger.error(f"Container failed: {str(e)}")
                return False
            
            time.sleep(check_interval)
            attempts += 1
            if attempts % 1 == 0:
                logger.info(f"Waiting for server to be ready... ({int(time.time() - start_time)}s)")
        
        # If we timeout, get container logs to help with debugging
        subprocess.run(f"docker logs {BENCHMARKING_CONTAINER_NAME}", shell=True)
        return False

    def run(
        self,
        model: Model, 
        backend_type: str, 
        hardware_type: str,
        no_weights: Optional[bool] = True
    ) -> tuple[bool, Optional[str]]:
        """
        Run the backend docker container based on model and HuggingFace token parameters.

        Args:
            model (str): The model identifier to use
            backend_type (str): Type of backend to run ('vllm' or 'tgi')
            no_weights (bool): Whether to download the model without weights. Defaults to True

        Returns:
            bool: True if the backend is working, False otherwise
        """
        try:
            # Validate backend type
            if backend_type not in get_backend_types():
                raise ValueError("backend_type must be either 'vllm' or 'tgi'")
            
            # Stop any existing containers
            try:
                subprocess.run("docker ps | grep vllm/vllm-openai | awk '{print $1}' | xargs -r docker stop", 
                            shell=True, check=True)
            except subprocess.CalledProcessError:
                pass  # Ignore errors if no containers are running
            
            hardware_info = get_hardware_info(hardware_type)
            
            # Read the appropriate template file
            template_path = f"src/backend/{backend_type}/{backend_type}.jinja2"
            with open(template_path, "r") as f:
                template_content = f.read()
                        
            if hardware_info.hardware_type == "default_settings":
                logger.warning("No hardware type specified, using default settings, this could lead to errors as some backends require to specify the hardware type")
                
            backend_info = hardware_info.backends[backend_type].model_dump(exclude_none=True)
            
            # Create template and render with parameters
            template = Template(template_content)
            
            extra_docker_args = {
                "hf_token": f"--env HF_TOKEN={get_token()}",
                "container_name": f"--name {BENCHMARKING_CONTAINER_NAME}",
            }
            
            if no_weights:
                extra_docker_args["hf_cache"] = f"--env HF_HUB_CACHE={NO_WEIGHTS_CACHE_DIR}"
            
            backend_info["benchmark_docker_args"] = f"{backend_info['docker_args']} {' '.join(extra_docker_args.values())}"
            del backend_info["docker_args"]

            template_args = {
                **backend_info
            }
            
            if backend_type == "llama_cpp":
                template_args["model"] = model.gguf_hf_model_id
                print("HF_HOME value:", HF_HOME)
                llama_cpp_cache_dir = os.path.join(HF_HOME, "llama.cpp")
                print("llama_cpp_cache_dir", llama_cpp_cache_dir)
                os.makedirs(llama_cpp_cache_dir, exist_ok=True)
                # Add home directory to template args
            elif backend_type == "tgi" or backend_type == "vllm":
                template_args["model"] = model.hf_model_id
            else:
                raise ValueError(f"Invalid backend type: {backend_type}")
            
            template_args["home_dir"] = str(Path.home())
            docker_command = template.render(**template_args)

            # Log the docker command (without sensitive info)
            token = get_token()
            if token:
                logger.info(f"Running docker command: {docker_command.replace(token, '[HIDDEN_TOKEN]')}")
            else:
                logger.info(f"Running docker command: {docker_command}")

            env = {}
            if no_weights:
                env["HUGGING_FACE_HUB_TOKEN"] = get_token()
                env["HF_HUB_CACHE"] = NO_WEIGHTS_CACHE_DIR

            # Execute the docker command in the background
            self.process = subprocess.Popen(
                docker_command, 
                shell=True, 
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start threads to log stdout and stderr
            stdout_thread = threading.Thread(target=self._log_stream, args=(self.process.stdout,))
            stderr_thread = threading.Thread(target=self._log_stream, args=(self.process.stderr,))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Check if process failed immediately
            time.sleep(2)  # Give it a moment to potentially fail
            if self.process.poll() is not None:
                # Process has already exited
                exit_code = self.process.poll()
                stderr_output = self.process.stderr.read()
                raise RuntimeError(f"Docker container failed to start (exit code {exit_code}): {stderr_output}")
            
            # Wait for server to be ready
            logger.info("Starting server and waiting for it to be ready...")
            if not self.wait_for_server(backend_type):
                raise RuntimeError("Server failed to start within timeout period or container exited")
            
            logger.info("Server is ready!")
            
        except Exception as e:
            logger.error(f"Failed to run docker container: {str(e)}")
            self.stop()  # Ensure cleanup on failure
            return False, None
        
        
        def clean_docker_command(docker_command: str) -> str:
            # First, join the lines and remove all backslashes and extra whitespace
            docker_command = docker_command.replace("\\\n", " ").strip()
            docker_command = " ".join(docker_command.split())
            
            # Replace token
            token = get_token()
            if token:
                docker_command = docker_command.replace(token, "YOUR_HF_TOKEN")
            
            # Replace home dir with ~
            home_dir = str(Path.home())
            docker_command = docker_command.replace(home_dir, "~")
            
            # Remove --name llm-hardware-benchmark
            docker_command = docker_command.replace("--name llm-hardware-benchmark", "")
            
            return docker_command.strip()
        
        cleaned_docker_command = clean_docker_command(docker_command)
        
        return True, cleaned_docker_command

    def stop(self):
        """Stop the running container and the associated process if they exist."""
        try:
            if self.process:
                self.process.kill()
                self.process = None
        except Exception as e:
            logger.error(f"Failed to kill process: {str(e)}")
        
        try:
            subprocess.run(f"docker rm -f {BENCHMARKING_CONTAINER_NAME}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to kill container: {str(e)}")
