# Dockerfile
FROM python:3.11-slim

# Configuración para ver logs inmediatamente
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar utilidades básicas del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copiar dependencias e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar el código fuente de la aplicación
# Esto toma la carpeta 'app' local y la pone en /app/app del contenedor
COPY app ./app

# Exponer el puerto de la API
EXPOSE 8081

# Comando de arranque apuntando a la app unificada
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]