# Base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# slow rebuild, need to build manually (build 1 time first)
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Copy requirements (requirements belong code)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run app
#CMD ["uvicorn", "App.Main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD ["python", "-u", "-m", "uvicorn", "App.Main:app", "--host", "0.0.0.0", "--port", "8000"]
