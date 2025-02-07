# from abc import ABC, abstractmethod
# from pydantic import BaseModel


# class BenchmarkResult(BaseModel):
#     model_id: str
#     time_to_first_token_secs: float
#     token_throughput_secs: float


# class BenchmarkParser(ABC):
#     @abstractmethod
#     def parse(self, file_path: str) -> BenchmarkResult:
#         pass
