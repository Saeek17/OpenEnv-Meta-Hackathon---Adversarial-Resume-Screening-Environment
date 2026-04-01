# Use python:3.11-slim as base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project context
COPY . .

# Expose port 8000
EXPOSE 8000

# Run uvicorn server.app:app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
