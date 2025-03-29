# ML Model Deployment with Flask and Docker

## Overview
This project demonstrates the deployment of machine learning models using a Flask API. It integrates two machine learning tasks into a single application:

1. **Classification Model (Iris Dataset):**
   - **Model:** A RandomForestClassifier trained on the Iris dataset.
   - **Input Requirement:** Exactly 4 float values per sample.
   - **Endpoints:**
     - **GET /predict/classification:** Returns sample predictions (for multiple inputs) without confidence values.
     - **POST /predict/classification:** Accepts JSON input and returns:
       - For a single sample: prediction with confidence.
       - For multiple samples: an array of predictions.

2. **Regression Model (Housing Dataset):**
   - **Model:** A RandomForestRegressor trained on a housing dataset.
   - **Input Requirement:** Exactly 20 float values per sample.
   - **Endpoints:**
     - **GET /predict/regression:** Returns sample predictions (with confidence values).
     - **POST /predict/regression:** Accepts JSON input and returns:
       - For a single sample: prediction with confidence.
       - For multiple samples: arrays of predictions and confidence values.

Other endpoints include:
- **GET /health:** Returns a simple health status.
- **GET /dashboard:** Serves a web-based dashboard UI.

## Setup Steps

1. **Clone the Repository:**
   ```bash
   git clone <https://github.com/Phithatsanan/MLModelWithDocker.git>
   cd <MLModelWithDocker>
   ```

2. **Install Dependencies:** Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models:**
   - Classification Model:
     ```bash
     python train.py
     ```
     This will generate a file named `model.pkl`.
   - Regression Model:
     ```bash
     python train_regression.py
     ```
     This will generate a file named `reg_model.pkl`.

4. **Run the Flask API:**
   ```bash
   python app.py
   ```
   The API will be available at http://localhost:9000.

5. **(Optional) Docker Deployment:**
   - Build the Docker image:
     ```bash
     docker build -t ml-model .
     ```
   - Run the Docker container:
     ```bash
     docker run -p 9000:9000 ml-model
     ```

## API Endpoints

### Health Check
- **Endpoint:** GET /health
- **Description:** Returns a simple JSON object with the health status.
- **Sample Request:**
  ```bash
  curl http://localhost:9000/health
  ```
- **Sample Response:**
  ```json
  { "status": "ok" }
  ```

### Classification Prediction

#### Single Input
- **Endpoint:** POST /predict/classification
- **Input Example:**
  ```json
  { "features": [5.1, 3.5, 1.4, 0.2] }
  ```
- **Expected Output Example:**
  ```json
  { "prediction": 0, "confidence": 0.97 }
  ```

#### Multiple Inputs
- **Endpoint:** POST /predict/classification
- **Input Example:**
  ```json
  {
    "features": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 3.4, 5.4, 2.3]
    ]
  }
  ```
- **Expected Output Example:**
  ```json
  { "predictions": [0, 2] }
  ```

#### GET Request:
If you send a GET request to `/predict/classification`, it returns sample predictions for multiple inputs (without confidence values).

### Regression Prediction

#### Single Input
- **Endpoint:** POST /predict/regression
- **Input Example:**
  ```json
  {
    "features": [7420, 4, 2, 3, 2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
  }
  ```
- **Expected Output Example:**
  ```json
  { "prediction": 300000.0, "confidence": 0.95 }
  ```

#### Multiple Inputs
- **Endpoint:** POST /predict/regression
- **Input Example:**
  ```json
  {
    "features": [
      [7420, 4, 2, 3, 2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
      [1500, 2, 1, 20, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    ]
  }
  ```
- **Expected Output Example:**
  ```json
  {
    "predictions": [300000.0, 150000.0],
    "confidences": [0.95, 0.90]
  }
  ```

#### GET Request:
Sending a GET request to `/predict/regression` returns sample predictions with confidence values.

### Dashboard UI
Open your web browser and navigate to:
```
http://localhost:9000/dashboard
```
This dashboard provides a user-friendly interface for entering inputs and viewing predictions for both classification and regression models.