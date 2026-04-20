# Use python:3.11-slim as base
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project context (including models.py and server/)
COPY . .

# Set PYTHONPATH to include the current directory so server can find models
ENV PYTHONPATH=/app

# Expose port 7860
EXPOSE 7860

# Run the Fleet environment (v3.0.0 — multi-agent hiring fleet)
CMD ["uvicorn", "server.fleet_app:app", "--host", "0.0.0.0", "--port", "7860"]
