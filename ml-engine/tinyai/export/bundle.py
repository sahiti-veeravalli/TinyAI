import os
import json
import shutil
import pickle
from typing import Dict, Any, List, Optional
import numpy as np


class DeploymentPackager:
    """Packages a selected compressed model into a standalone, reproducible production edge deployment bundle."""

    @classmethod
    def generate_bundle(
        cls,
        bundle_dir: str,
        model_artifact_path: str,
        preprocessor: Any,
        feature_names: List[str],
        model_name: str,
        model_family: str,
        quality_score: float,
        latency_ms: float,
        disk_size_kb: float,
        target_column: str = "target",
        sample_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        os.makedirs(bundle_dir, exist_ok=True)

        # 1. Save compressed model artifact
        dest_model_path = os.path.join(bundle_dir, "model.pkl")
        if os.path.exists(model_artifact_path):
            shutil.copy2(model_artifact_path, dest_model_path)
        else:
            raise FileNotFoundError(f"Model artifact not found at {model_artifact_path}")

        # 2. Save preprocessor pipeline
        dest_prep_path = os.path.join(bundle_dir, "preprocessor.pkl")
        with open(dest_prep_path, "wb") as f:
            pickle.dump(preprocessor, f)

        # 3. Save feature schema
        schema_path = os.path.join(bundle_dir, "input_schema.json")
        schema_data = {
            "feature_names": feature_names,
            "numeric_columns": getattr(preprocessor, "numeric_cols", []),
            "categorical_columns": getattr(preprocessor, "categorical_cols", []),
            "target_column": target_column
        }
        with open(schema_path, "w") as f:
            json.dump(schema_data, f, indent=2)

        # 4. Generate Model Card
        card_path = os.path.join(bundle_dir, "model_card.json")
        card_data = {
            "model_name": model_name,
            "model_family": model_family,
            "target_column": target_column,
            "performance": {
                "quality_score": quality_score,
                "latency_p50_ms": latency_ms,
                "disk_size_kb": disk_size_kb
            },
            "edge_runtime_targets": ["Linux x86_64", "ARM64 (Raspberry Pi / Jetson)", "Bare-metal container"],
            "license": "Apache-2.0",
            "intended_use": "High-throughput edge inferencing with strict resource budgets."
        }
        with open(card_path, "w") as f:
            json.dump(card_data, f, indent=2)

        # 5. Generate FastAPI Inference Microservice (`app.py`)
        app_code = f'''"""
TinyAI Standalone Edge Inference Service
Model: {model_name} ({model_family})
Quality: {quality_score:.4f} | Latency: {latency_ms:.3f}ms | Size: {disk_size_kb:.1f}KB
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, create_model

app = FastAPI(
    title="TinyAI Edge Inference Service - {model_name}",
    version="1.0.0",
    description="Edge optimized production inference microservice."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

# Load model and preprocessor at startup
model_obj = None
preprocessor = None

@app.on_event("startup")
def load_artifacts():
    global model_obj, preprocessor
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            model_obj = data.get("model", data) if isinstance(data, dict) else data

        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        print("Successfully loaded model and preprocessor artifacts.")
    except Exception as e:
        print(f"Error loading model artifacts: {{e}}")
        raise e

class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]] = None
    latency_ms: float
    model_name: str = "{model_name}"

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {{"status": "healthy", "model_loaded": model_obj is not None}}

@app.get("/ready", status_code=status.HTTP_200_OK)
def readiness_check():
    if model_obj is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded")
    return {{"status": "ready"}}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model_obj is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    t0 = time.perf_counter()
    try:
        df = pd.DataFrame(request.records)
        X = preprocessor.transform(df)
        preds = model_obj.predict(X).tolist()

        probs = None
        if hasattr(model_obj, "predict_proba"):
            p = model_obj.predict_proba(X)
            if p.ndim == 2 and p.shape[1] == 2:
                probs = p[:, 1].tolist()
            else:
                probs = np.max(p, axis=1).tolist()

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionResponse(
            predictions=preds,
            probabilities=probs,
            latency_ms=round(lat_ms, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {{str(e)}}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''
        app_path = os.path.join(bundle_dir, "app.py")
        with open(app_path, "w") as f:
            f.write(app_code)

        # 6. Generate Dockerfile
        dockerfile_code = """FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
"""
        dockerfile_path = os.path.join(bundle_dir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_code)

        # 7. Generate requirements.txt
        reqs_code = """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.7.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
xgboost>=2.0.0
"""
        req_path = os.path.join(bundle_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write(reqs_code)

        # 8. Generate load_test.py
        load_code = f"""import time
import requests
import json

SERVER_URL = "http://localhost:8080/predict"
sample_record = {json.dumps(sample_input or {f: 1.0 for f in getattr(preprocessor, 'numeric_cols', ['f1', 'f2'])})}

payload = {{"records": [sample_record] * 10}}

print("Starting load test against TinyAI Edge Server...")
successes = 0
total_time = 0.0

for i in range(50):
    t0 = time.perf_counter()
    res = requests.post(SERVER_URL, json=payload)
    elapsed = (time.perf_counter() - t0) * 1000.0
    if res.status_code == 200:
        successes += 1
        total_time += elapsed

print(f"Completed 50 batch requests (500 records).")
print(f"Success rate: {{(successes / 50) * 100:.1f}}%")
print(f"Average Round-Trip Latency: {{(total_time / successes):.2f}} ms")
"""
        load_path = os.path.join(bundle_dir, "load_test.py")
        with open(load_path, "w") as f:
            f.write(load_code)

        # 9. Deployment README.md
        readme_code = f"""# Edge Inference Service: {model_name}

Optimized with **TinyAI Edge Intelligence Platform**.

## Model Specs
- **Model Family**: `{model_family}`
- **Quality Score**: `{quality_score:.4f}`
- **Latency (p50)**: `{latency_ms:.3f} ms`
- **Compressed Size**: `{disk_size_kb:.1f} KB`

## Quick Start (Local Run)
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Run with Docker
```bash
docker build -t tinyai-{model_family.lower()}:latest .
docker run -p 8080:8080 tinyai-{model_family.lower()}:latest
```

## Verify Health
```bash
curl http://localhost:8080/health
```

## Run Prediction
```bash
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"records": [{json.dumps(sample_input or {})}]}}'
```
"""
        readme_path = os.path.join(bundle_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_code)

        return {
            "model_path": dest_model_path,
            "preprocessor_path": dest_prep_path,
            "app_path": app_path,
            "dockerfile_path": dockerfile_path,
            "requirements_path": req_path,
            "model_card_path": card_path,
            "readme_path": readme_path,
            "schema_path": schema_path
        }
