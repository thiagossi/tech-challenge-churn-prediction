# Variáveis para os comandos do Makefile
PYTHON = python
PIP = pip
UVICORN = uvicorn
SRC_DIR = src
TESTS_DIR = tests

.PHONY: help install lint format test run train clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make install  - Instala as dependências do projeto"
	@echo "  make lint     - Executa a verificação de estilo com Ruff"	
	@echo "  make test     - Executa a suíte de testes automatizados"
	@echo "  make run      - Inicia a API FastAPI localmente"
	@echo "  make train    - Executa o script de treinamento do modelo"
	@echo "  make clean    - Remove arquivos temporários e caches"

install:
	$(PIP) install -e .

lint:
	ruff check .

test:
	$(PYTHON) -m pytest $(TESTS_DIR) -v --cov=src --cov-report=term-missing

run:
	$(UVICORN) src.main:app --reload

train:
	$(PYTHON) src/train.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +