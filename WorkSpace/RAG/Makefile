.PHONY: start-all stop-all run-ui run-api run-all-local status

ROOT := $(CURDIR)

start-all:
	@echo "Starting all services via Docker Compose" 
	@docker-compose -f $(ROOT)/docker-compose.yml up --build -d

stop-all:
	@echo "Stopping all services" 
	@docker-compose -f $(ROOT)/docker-compose.yml down

run-ui:
	@echo "Starting UI (Streamlit)" 
@docker-compose -f $(ROOT)/docker-compose.yml up -d ui

run-api:
	@echo "Starting API (FastAPI)" 
@docker-compose -f $(ROOT)/docker-compose.yml up -d api

run-all-local:
	@echo "Starting UI and API locally (no Docker)" 
@python $(ROOT)/run_all.py --mode local

status:
	@docker-compose -f $(ROOT)/docker-compose.yml ps
