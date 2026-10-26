ifeq ($(OS),Windows_NT)
	VENV_PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv\Scripts\python.exe,py -3.11)
	VENV_PIP := $(if $(wildcard .venv/Scripts/pip.exe),.venv\Scripts\pip.exe,py -3.11 -m pip)
else
	VENV_PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
	VENV_PIP := $(if $(wildcard .venv/bin/pip),.venv/bin/pip,pip3)
endif

.PHONY: setup data train eval test test-model test-perf lint cov run map

setup:
	python -m venv .venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt
	$(VENV_PIP) install -e . --no-deps
	$(VENV_PYTHON) scripts/make_fixtures.py

data:
	$(VENV_PYTHON) -m rdai.data.download
	$(VENV_PYTHON) -m rdai.data.voc_to_yolo
	$(VENV_PYTHON) -m rdai.data.split

train:
	$(VENV_PYTHON) -m rdai.train.train

eval:
	$(VENV_PYTHON) -m rdai.train.evaluate
	$(VENV_PYTHON) scripts/make_results.py

test:
	$(VENV_PYTHON) -m pytest -m "not slow and not gpu and not perf"

test-model:
	$(VENV_PYTHON) -m pytest tests/model -m "slow or gpu"

test-perf:
	$(VENV_PYTHON) -m pytest tests/perf -m "perf"

lint:
	$(VENV_PYTHON) -m ruff check src tests scripts
	$(VENV_PYTHON) -m ruff format --check src tests scripts

cov:
	$(VENV_PYTHON) -m pytest -m "not slow and not gpu and not perf" --cov=src/rdai --cov-fail-under=80

run:
	$(VENV_PYTHON) -m uvicorn rdai.api.main:app --host 0.0.0.0 --port 8000 --reload

map:
	$(VENV_PYTHON) scripts/make_map.py
