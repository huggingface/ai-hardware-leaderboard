from typing import Optional, List
import yaml
from pydantic import BaseModel, ValidationError
import os

class BackendInfo(BaseModel):
    docker_args: Optional[str]

class HardwareInfo(BaseModel):
    hardware_type: str
    backends: dict[str, BackendInfo]

def get_hardware_info() -> HardwareInfo:
    with open("src/hardware/hardware_info.yaml", "r") as f:
        hardware_info_data = yaml.load(f, Loader=yaml.FullLoader)
    
    # Validate and parse the loaded YAML data
    if not isinstance(hardware_info_data, list):
        raise ValueError("Invalid hardware_info format: expected a list of hardware configurations")
    
    hardware_infos: List[HardwareInfo] = []
    for item in hardware_info_data:
        try:
            hardware_infos.append(HardwareInfo(**item))
        except ValidationError as e:
            raise ValueError(f"Invalid hardware_info format: {str(e)}")
    
    hardware_info = os.environ.get("HARDWARE_TYPE", "default_settings")
    hardware_info = next((info for info in hardware_infos if info.hardware_type == hardware_info), None)
    if hardware_info is None:
        raise ValueError(f"Hardware type '{hardware_info}' not found in hardware_info.yaml")
    
    return hardware_info