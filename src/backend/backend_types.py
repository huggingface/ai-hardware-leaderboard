from enum import Enum

class BackendType(str, Enum):
    VLLM = "vllm"
    TGI = "tgi"


def get_backend_types() -> list[str]:
    """
    Get the list of backend types, for example ['vllm', 'tgi']
    """
    return [backend.value for backend in BackendType]