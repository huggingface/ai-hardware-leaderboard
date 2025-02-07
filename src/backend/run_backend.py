from jinja2 import Template
import subprocess
import time
import threading
from typing import Optional
from weights import NO_WEIGHTS_CACHE_DIR
from huggingface_hub.utils._auth import get_token
from hardware.hardware_info import get_hardware_info
from backend.backend_types import get_backend_types
from loguru import logger
import requests


class BackendRunner:
    BENCHMARKING_CONTAINER_NAME = "llm-hardware-benchmark"

    def __init__(self):
        self.process = None

    @staticmethod
    def _log_stream(stream, prefix):
        """Log output from a stream line by line with a prefix."""
        for line in iter(stream.readline, ''):
            if line:
                logger.info(f"{prefix}: {line.strip()}")

    def wait_for_server(self, timeout=3000, check_interval=10):
        """
        Wait for the server to be ready by checking the health endpoint.
        
        Args:
            timeout (int): Maximum time to wait in seconds
            check_interval (int): Time between checks in seconds
        
        Returns:
            bool: True if server is ready, False otherwise
        """   
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get('http://localhost:8080/health')
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(check_interval)
            logger.info(f"Waiting for server to be ready... ({int(time.time() - start_time)}s)")
        return False

    def run(
        self,
        model: str, 
        backend_type: str, 
        no_weights: Optional[bool] = True
    ):
        """
        Run the backend docker container based on model and HuggingFace token parameters.

        Args:
            model (str): The model identifier to use
            backend_type (str): Type of backend to run ('vllm' or 'tgi')
            no_weights (bool): Whether to download the model without weights. Defaults to True
        """
        # Validate backend type
        if backend_type not in get_backend_types():
            raise ValueError("backend_type must be either 'vllm' or 'tgi'")
        
        # Stop any existing containers
        try:
            subprocess.run("docker ps | grep vllm/vllm-openai | awk '{print $1}' | xargs -r docker stop", 
                        shell=True, check=True)
        except subprocess.CalledProcessError:
            pass  # Ignore errors if no containers are running
        
        hardware_info = get_hardware_info()
        
        # Read the appropriate template file
        template_path = f"src/backend/{backend_type}.jinja2"
        with open(template_path, "r") as f:
            template_content = f.read()
            
        docker_args = hardware_info.backends[backend_type].docker_args
        if docker_args is None:
            docker_args = ""

        # Create template and render with parameters
        template = Template(template_content)
        
        template_args = {
            "model": model,
            "hf_token": get_token(),
            "docker_args": docker_args,
            "container_name_arg": f"--name {self.BENCHMARKING_CONTAINER_NAME}"
        }
        
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
        try:
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
            stdout_thread = threading.Thread(target=self._log_stream, args=(self.process.stdout, "DOCKER"))
            stderr_thread = threading.Thread(target=self._log_stream, args=(self.process.stderr, "DOCKER ERROR"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for server to be ready
            logger.info("Starting server and waiting for it to be ready...")
            if not self.wait_for_server():
                self.process.kill()
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                raise RuntimeError("Server failed to start within timeout period")
            
            logger.info("Server is ready!")
            
        except Exception as e:
            logger.error(f"Failed to run docker container: {str(e)}")
            raise

    def stop(self):
        """Stop the running container and the associated process if they exist."""
        try:
            if self.process:
                self.process.kill()
                self.process = None
        except Exception as e:
            logger.error(f"Failed to kill process: {str(e)}")
        
        try:
            subprocess.run(f"docker rm -f {self.BENCHMARKING_CONTAINER_NAME}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to kill container: {str(e)}")
