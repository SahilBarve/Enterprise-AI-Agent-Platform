.PHONY: help install up down restart logs test lint format typecheck clean

help:
	@echo "Enterprise AI Operations Platform - Development Commands"
	@echo "  make install    Install local editable package and dev dependencies"
	@echo "  make up         Start local infrastructure stack via Docker Compose"
	@echo "  make down       Stop local infrastructure stack"
	@echo "  make restart    Restart local infrastructure stack"
	@echo "  make logs       Tail docker compose logs"
	@echo "  make test       Run all unit tests with coverage"
	@echo "  make lint       Run Ruff linter and formatter checks"
	@echo "  make format     Format code with Ruff"
	@echo "  make typecheck  Run Mypy static type analysis"
	@echo "  make clean      Remove caches, temp files, and artifacts"

install:
	pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

test:
	pytest tests/unit -v --cov=libs --cov=services

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy libs services

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
