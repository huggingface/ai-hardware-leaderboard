# LLM-Hardware-Leaderboard

# How to use

please select your hardware type

cp .env.example .env

uv venv
source .venv/bin/activate
uv pip install -e .

- start different backend
- download with no weight

gradio interface
- compatibility as first link with each hardware type

- document for hardware partner to help give us hardware to run our leaderboard
-> the philiosphy no manual ops or anything docker run that serve a openai container request
-> no crazy docker argument (meaning sensible default)
What is benefit you show your hardware support the latest and greatest
-> need to be in TGI or vLLM (or strong community demand for another backend such a llama.cpp, like support on edge device (or should i do ollama directly)) cannot expected user to learn a new framework
-> contribute hardware to run nightly ci on it

-> simpler only a single request

hardware to support (use generic name and then give more info about the specific hardware)
- nvidia gpu (done)
- intel cpu (in-working)
- intel habana
- intel gpu
- google tpu
- aws inferentia
- amd gpu
- amd cpu
- apple silicon cpu
(use their logo)
-> simple green box if compatible 

- save the command line to allow people to run the container on the platform (give more credibility)

Later:
- montior energy with code-carbon
- montior memory usage
- dashboard to get more info like the running of the each benchmark (can i do all in 6h? otherwise max it parrallel on each model so 10 jobs? -> faster)
- performance for each type of hardware (with aggregate option)
- startup time (need to do docker pull first then)
- find a way to get price


explain philosphy behind it:
why benchamrk llm, why openai, big use cases


-> need to think about how to be smart about the benchmarks

if out of memory should not count as failure...?

also do not use gguf models?


how to manually test a backend with curl

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{
    "model": "openai-community/gpt2",
    "messages": [
      {
        "role": "user",
        "content": "What is Deep Learning?"
      }
    ],
    "max_tokens": 20
  }'


  - volume should be optional to prevent filling up the harddrive


docker run --gpus=all -p 8080:11434 --name llm-hardware-benchmark ollama/ollama

docker exec -it llm-hardware-benchmark ollama run llama3

curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-X POST \
-d '{
"model": "llama3.2:1b",
"messages": [
    {
    "role": "user",
    "content": "What is Deep Learning?"
    }
],
"max_tokens": 20
}'



ollama pull llama3
- make this a tui instead

- add status page on ai model is there are working 