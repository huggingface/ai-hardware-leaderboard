style:
	ruff format .
	ruff check --fix .

kill-port:
	kill -9 $(lsof -t -i:8080)

run-benchmark:
	llm-hardware