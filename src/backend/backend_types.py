from enum import Enum

class BackendType(str, Enum):
    TGI = "tgi"
    # VLLM = "vllm"
    # LLAMA_CPP = "llama_cpp"


def get_backend_types() -> list[str]:
    """
    Get the list of backend types, for example ['vllm', 'tgi']
    """
    return [backend.value for backend in BackendType]