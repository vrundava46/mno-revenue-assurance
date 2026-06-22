# MNO Revenue Assurance — developer commands
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export TELCO_RAG_FORCE_FALLBACK ?= 0

.PHONY: help setup setup-embeddings generate warehouse index pipeline query investigate test ui clean

help:
	@echo "make setup        - create .venv and install dependencies"
	@echo "make pipeline     - generate data -> build warehouse -> index docs"
	@echo "make query SQL=   - run a read-only SQL query on the warehouse"
	@echo "make investigate Q= - run the agent on an investigation question"
	@echo "make ui           - launch the Streamlit app"
	@echo "make test         - run the test suite (offline, deterministic)"
	@echo "make clean        - remove generated data, warehouse & vectors"

setup:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo "Done. Optional: 'make setup-embeddings' and './setup_ollama.sh'."

setup-embeddings:
	$(PIP) install -q -r requirements-embeddings.txt

generate:
	$(PY) -m ra_agent.app.cli generate

warehouse:
	$(PY) -m ra_agent.app.cli warehouse

index:
	$(PY) -m ra_agent.app.cli index

pipeline:
	$(PY) -m ra_agent.app.cli pipeline

query:
	$(PY) -m ra_agent.app.cli query "$(SQL)"

investigate:
	$(PY) -m ra_agent.app.cli investigate "$(Q)"

test:
	TELCO_RAG_FORCE_FALLBACK=1 $(PY) -m pytest -q

ui:
	$(VENV)/bin/streamlit run ra_agent/app/streamlit_app.py

clean:
	rm -rf data_store
