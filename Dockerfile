FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_minimal ./rag_minimal
COPY docs ./docs
COPY README.md .

EXPOSE 8501

CMD ["streamlit", "run", "rag_minimal/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
