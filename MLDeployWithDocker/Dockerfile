# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 9000 for the Flask API
EXPOSE 9000

# Run the combined Flask API that serves both models and the UI
CMD ["python", "app.py"]
