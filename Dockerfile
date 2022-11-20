FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim
RUN pip install --no-cache-dir transformers==4.1.1 tensorflow==2.9.1 numpy==1.23.1 pydantic==1.9.1
COPY main.py ./main.py
COPY ../sentiment /sentiment