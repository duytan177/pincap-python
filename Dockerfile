# Base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements
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
