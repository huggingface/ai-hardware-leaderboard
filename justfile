run-benchmark:
	uv run ai-hardware-leaderboard 

style:
	ruff format .
	ruff check --fix .

kill-port:
	kill -9 $(lsof -t -i:8080)

copy-env:
	cp .env.example .env
