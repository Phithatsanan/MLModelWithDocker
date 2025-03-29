# ML Model Deployment with Docker

## Overview
This project demonstrates the deployment of machine learning models using Flask and Docker. It integrates two parts into a single Flask application:

1. **Classification Model:**
   - **Model:** A RandomForestClassifier trained on the Iris dataset (expects 4 features).
   - **Endpoint:**  
     - **POST /predict/classification:**  
       - *Single Input Example:*  
         - **Input:**  
           ```json
           { "features": [5.1, 3.5, 1.4, 0.2] }
           ```
         - **Response:**  
           ```json
           { "prediction": 0, "confidence": 0.97 }
           ```
       - *Multiple Inputs Example:*  
         - **Input:**  
           ```json
           { "features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]] }
           ```
         - **Response:**  
           ```json
           { "predictions": [0, 2] }
           ```
   - **Additional Endpoints:**  
     - **GET /health:** Returns `{ "status": "ok" }`.  
     - **GET /dashboard:** Serves the input/output dashboard.

2. **Regression Model:**
   - **Model:** A RandomForestRegressor trained on a housing dataset (`Housing.csv`) to predict housing prices.
   - **Endpoint:**  
     - **POST /predict/regression:**  
       - **Input Example:**  
         ```json
         { "features": [2100, 3, 2, 10, ...] }
         ```
       - **Response Example:**  
         ```json
         { "prediction": 300000.0, "confidence": 0.95 }
         ```
       - *Note:* The number of features must match the regression model's expected input (after any preprocessing).

## Files
- **train.py:** Trains the classification model (Iris) and saves it as `model.pkl`.
- **train_regression.py:** Trains the regression model (Housing) and saves it as `reg_model.pkl`.
- **app.py:** Combined Flask API for both classification and regression predictions, health check, and dashboard.
- **static/index.html:** Dashboard user interface for input and output.
- **Dockerfile:** Docker configuration to containerize the application.
- **requirements.txt:** Lists all dependencies.
- **README.md:** Project documentation.
- **Housing.csv:** The housing dataset (place this file in the project directory).

## Setup and Running

### 1. Train the Models
**Classification Model:**
```bash
python train.py
```
This generates `model.pkl`.

**Regression Model:**
```bash
python train_regression.py
```
This generates `reg_model.pkl`.

### 2. Running the API Locally
Run the combined Flask API:
```bash
python app.py
```
The API will be available on port 9000:
- Health Check: `http://localhost:9000/health`
- Dashboard: `http://localhost:9000/dashboard`
- Classification Prediction: POST to `http://localhost:9000/predict/classification`
- Regression Prediction: POST to `http://localhost:9000/predict/regression`

### 3. Docker Deployment
**Build the Docker image:**
```bash
docker build -t ml-model .
```

**Run the Docker container:**
```bash
docker run -p 9000:9000 ml-model
```

## API Examples

### Health Check
```bash
curl http://localhost:9000/health
```
**Expected response:**
```json
{ "status": "ok" }
```

### Classification Prediction (Single Input)
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
     http://localhost:9000/predict/classification
```
**Example response:**
```json
{ "prediction": 0, "confidence": 0.97 }
```

### Classification Prediction (Multiple Inputs)
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}' \
     http://localhost:9000/predict/classification
```
**Example response:**
```json
{ "predictions": [0, 2] }
```

### Regression Prediction
*(Ensure the input has the correct number of features for your regression model.)*
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features": [<your housing model feature values> ]}' \
     http://localhost:9000/predict/regression
```
**Example response:**
```json
{ "prediction": 300000.0, "confidence": 0.95 }
```

## Dashboard Usage

Open your browser and navigate to:
```
http://localhost:9000/dashboard
```

The dashboard provides:
- A Health Check button.
- A section for Classification Prediction (enter exactly 4 numbers).
- A section for Regression Prediction (enter your housing model's feature values).