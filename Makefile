test: 
	python -m pytest tests/ -v

format:
	ruff format . 

lint:
	ruff check . --fix

docker-up:
	docker compose -f docker/docker-compose.yml up -d

setup:
	docker compose -f docker/docker-compose.yml up -d
	cd infraestructure && terraform apply -auto-approve
