FROM python:3.10-slim

# The 'Clean & Fix' approach to avoid Error 100
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Standard project setup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]