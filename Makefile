.PHONY: install dev test lint serve smoke docker-build docker-run kb-ingest clean

PY ?= python
PORT ?= 8000

install:
	$(PY) -m pip install -r requirements.txt

dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port $(PORT) --reload

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port $(PORT) --workers 1

smoke:
	$(PY) scripts/smoke.py

test:
	pytest -q

kb-ingest:
	$(PY) -m src.rag.ingest --src data/sample_kb --collection support

docker-build:
	docker build -t voice-rag-agent .

docker-run:
	docker run --rm -p $(PORT):8000 --env-file .env voice-rag-agent

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ */__pycache__ */*/__pycache__
	rm -rf .chroma
