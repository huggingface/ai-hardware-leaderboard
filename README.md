# LLM-Hardware-Leaderboard

# How to use

uv venv
source .venv/bin/activate
uv pip install .

- start different backend
- download with no weight

gradio interface
- compatibility as first link with each hardware type

- document for hardware partner to help give us hardware to run our leaderboard
-> the philiosphy no manual ops or anything docker run that serve a openai container request
-> no crazy docker argument (meaning sensible default)
What is benefit you show your hardware support the latest and greatest
-> need to be in TGI or vLLM (or strong community demand for another backend such a llama.cpp, like support on edge device) cannot expected user to learn a new framework
-> contribute hardware to run nightly ci on it

-> simpler only a single request

hardware to support (use generic name and then give more info about the specific hardware)
- nvidia gpu
- intel cpu
- intel habana
- intel gpu
- google tpu
- aws inferentia
- amd gpu
- amd cpu
(use their logo)
-> simple green box if compatible 

- save the command line to allow people to run the container on the platform (give more credibility)

Later:
- montior energy with code-carbon
- montior memory usage
- dashboard to get more info like the running of the each benchmark (can i do all in 6h? otherwise max it parrallel on each model so 10 jobs? -> faster)
- performance for each type of hardware (with aggregate option)
- startup time


explain philosphy behind it:
why benchamrk llm, why openai, big use cases


-> need to think about how to be smart about the benchmarks

if out of memory should not count as failure...?

also do not use gguf models?