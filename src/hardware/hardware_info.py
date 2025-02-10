from typing import Optional, List
import yaml
from pydantic import BaseModel, ValidationError, Extra
import os

class BackendInfo(BaseModel):
    docker_args: Optional[str] = None
    image_suffix: Optional[str] = None
    runtime_options: Optional[dict] = None
    image: Optional[str] = None
    
    class Config:
        extra = Extra.forbid

class HardwareInfo(BaseModel):
    hardware_type: str
    backends: dict[str, BackendInfo]
    
    class Config:
        extra = Extra.forbid

def get_hardware_info(hardware_type: str) -> HardwareInfo:
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
    
    hardware_info = next((info for info in hardware_infos if info.hardware_type == hardware_type), None)
    if hardware_info is None:
        raise ValueError(f"Hardware type '{hardware_type}' not found in hardware_info.yaml")
    
    return hardware_info