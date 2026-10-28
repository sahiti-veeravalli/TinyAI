.PHONY: all data train compress benchmark results test dev clean

all: benchmark

# Step 1: Generate or verify dataset bundles
data:
	python -c "from pipeline.datasets import load_benchmark_dataset; [load_benchmark_dataset(d) for d in ['credit_card_fraud', 'census_income', 'home_credit', 'bank_marketing']]"

# Step 2: Execute end-to-end benchmark & ablation suite
benchmark results:
	python scripts/generate_benchmarks.py

# Step 3: Run pipeline and API tests
test:
	pytest tests/ -v

# Step 4: Run local development servers
dev-backend:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting backend and frontend..."
	@echo "Backend on http://localhost:8000"
	@echo "Frontend on http://localhost:3000"

clean:
	rm -rf results/models/*.onnx
