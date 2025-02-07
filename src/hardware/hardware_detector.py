import subprocess
import os
from typing import List
from rich.console import Console
import platform

console = Console()

class HardwareDetector:
    def _run_cmd(self, cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('utf-8')
        except:
            return ""

    def detect_nvidia_gpu(self) -> bool:
        return bool(self._run_cmd(['nvidia-smi']))

    def detect_intel_cpu(self) -> bool:
        if platform.system() == "Linux":
            cpu_info = self._run_cmd(['cat', '/proc/cpuinfo'])
            return 'Intel' in cpu_info
        elif platform.system() == "Darwin":  # macOS
            sysctl_output = self._run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
            return 'Intel' in sysctl_output
        elif platform.system() == "Windows":
            wmic_output = self._run_cmd(['wmic', 'cpu', 'get', 'name'])
            return 'Intel' in wmic_output
        return False

    def detect_amd_cpu(self) -> bool:
        if platform.system() == "Linux":
            cpu_info = self._run_cmd(['cat', '/proc/cpuinfo'])
            return 'AMD' in cpu_info
        elif platform.system() == "Darwin":  # macOS
            sysctl_output = self._run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
            return 'AMD' in sysctl_output
        elif platform.system() == "Windows":
            wmic_output = self._run_cmd(['wmic', 'cpu', 'get', 'name'])
            return 'AMD' in wmic_output
        return False

    def detect_intel_gpu(self) -> bool:
        if platform.system() == "Linux":
            lspci_output = self._run_cmd(['lspci'])
            return 'Intel' in lspci_output and ('VGA' in lspci_output or 'Display' in lspci_output)
        elif platform.system() == "Darwin":
            system_profiler = self._run_cmd(['system_profiler', 'SPDisplaysDataType'])
            return 'Intel' in system_profiler
        return False

    def detect_amd_gpu(self) -> bool:
        if platform.system() == "Linux":
            lspci_output = self._run_cmd(['lspci'])
            return ('AMD' in lspci_output or 'Radeon' in lspci_output) and ('VGA' in lspci_output or 'Display' in lspci_output)
        elif platform.system() == "Darwin":
            system_profiler = self._run_cmd(['system_profiler', 'SPDisplaysDataType'])
            return 'AMD' in system_profiler or 'Radeon' in system_profiler
        return False

    def detect_habana(self) -> bool:
        return os.path.exists('/dev/habana_pci0')

    def detect_tpu(self) -> bool:
        return os.path.exists('/dev/accel0')

    def detect_inferentia(self) -> bool:
        return os.path.exists('/dev/neuron0')

    def detect_apple_silicon(self) -> bool:
        if platform.system() != "Darwin":
            return False
        cpu_brand = self._run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
        return 'Apple' in cpu_brand

    def get_recommended_hardware(self) -> List[str]:
        recommended = []
        
        if self.detect_nvidia_gpu():
            recommended.append(('cuda', 'NVIDIA GPU detected'))
        if self.detect_intel_cpu():
            recommended.append(('intel_cpu', 'Intel CPU detected'))
        if self.detect_amd_cpu():
            recommended.append(('amd_cpu', 'AMD CPU detected'))
        if self.detect_intel_gpu():
            recommended.append(('intel_gpu', 'Intel GPU detected'))
        if self.detect_amd_gpu():
            recommended.append(('rocm', 'AMD GPU detected'))
        if self.detect_habana():
            recommended.append(('habana', 'Habana device detected'))
        if self.detect_tpu():
            recommended.append(('tpu', 'Google TPU detected'))
        if self.detect_inferentia():
            recommended.append(('inferentia', 'AWS Inferentia detected'))
        if self.detect_apple_silicon():
            recommended.append(('apple_silicon', 'Apple Silicon detected'))

        return recommended