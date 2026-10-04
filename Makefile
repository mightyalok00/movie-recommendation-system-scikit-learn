.PHONY: help install test lint run-all build-artifacts api clean docker-build docker-up

help:
	@echo "Available commands:"
	@echo "  make install          Install dependencies"
	@echo "  make test             Run unit test suite"
	@echo "  make run-all          Execute all 11 solution sections"
	@echo "  make build-artifacts  Train and serialize deployment artifacts"
	@echo "  make api              Start FastAPI REST service"
	@echo "  make clean            Remove cache and temporary files"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-up        Start FastAPI in Docker"

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	python main.py test

run-all:
	python main.py run-all

build-artifacts:
	python main.py build-artifacts

api:
	python main.py api --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t movielens-32m .

docker-up:
	docker-compose up -d
