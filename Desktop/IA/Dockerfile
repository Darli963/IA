FROM python:3.12-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala las dependencias Python
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copia la carpeta api
COPY api /app/api

# Expone el puerto 8000
EXPOSE 8000

# Ejecuta uvicorn
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
