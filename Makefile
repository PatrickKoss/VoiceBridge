.PHONY: prepare lint test test-fast clean help

# Python and pip from venv
PYTHON = venv/bin/python
PIP = venv/bin/pip

help:
	@echo "Available commands:"
	@echo "  prepare     - Initialize virtual environment and install dependencies"
	@echo "  lint        - Run ruff linting and fix issues"
	@echo "  test        - Run all tests with coverage"
	@echo "  test-fast   - Run tests without coverage"
	@echo "  clean       - Clean up cache files and virtual environment"

prepare: venv/bin/activate

venv/bin/activate: requirements.txt
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ -f test_requirements.txt ]; then $(PIP) install -r test_requirements.txt; fi
	touch venv/bin/activate

lint: venv/bin/activate
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test: venv/bin/activate
	$(PYTHON) -m pytest --cov=. --cov-report=html --cov-report=term

test-fast: venv/bin/activate
	$(PYTHON) -m pytest

clean:
	rm -rf venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete