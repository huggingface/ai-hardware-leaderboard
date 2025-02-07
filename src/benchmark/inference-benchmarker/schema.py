# import json
# from ..benchmark_result import BenchmarkResult, BenchmarkParser


# class VLLMBenchmarkParser(BenchmarkParser):
#     def parse(self, model_id: str, file_path: str) -> BenchmarkResult:
#         with open(file_path, "r") as f:
#             data = json.load(f)

#         # Sanity check
#         assert data["config"]["tokenizer"] == model_id

#         # Find the "throughput" result section
#         throughput_result = next(
#             result for result in data["results"] if result["id"] == "throughput"
#         )

#         # Extract metrics
#         time_to_first_token = (
#             throughput_result["time_to_first_token_ms"]["avg"] / 1000
#         )  # Convert to seconds
#         token_throughput = throughput_result["token_throughput_secs"]

#         return BenchmarkResult(
#             model_id=model_id,
#             time_to_first_token_secs=time_to_first_token,
#             token_throughput_secs=token_throughput,
#         )
