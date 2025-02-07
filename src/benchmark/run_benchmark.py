# from jinja2 import Template
# import subprocess
# from huggingface_hub.utils._auth import get_token
# from weights import NO_WEIGHTS_CACHE_DIR

# def run_benchmark(model: str):
#     """
#     Run the benchmark docker container based on model and HuggingFace token parameters.

#     Args:
#         model (str): The model identifier to use
#         hf_token (str): HuggingFace authentication token
#     """
#     # Read the template file
#     template_path = "benchmark/inference-benchmarker.jinja2"
#     with open(template_path, "r") as f:
#         template_content = f.read()

#     # Create template and render with parameters
#     template = Template(template_content)
#     docker_command = template.render(model=model)

#     # Execute the docker command and capture output
#     subprocess.run(
#         docker_command,
#     )
