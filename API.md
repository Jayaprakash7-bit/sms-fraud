# SMS Fraud Detection API

This API provides endpoints for SMS fraud detection and chatbot functionality.

## Base URL
When deployed, the base URL will be provided by the hosting platform (e.g., `https://your-app-name.onrender.com`).

## Endpoints

### POST /api/predict
Predict if a single SMS message is fraudulent.

**Request Body:**
```json
{
  "text": "Your SMS message here"
}
```

**Response:**
```json
{
  "fraud_probability": 0.95,
  "is_fraud": true,
  "threshold": 0.5
}
```

**Error Responses:**
- 400: Empty message
- 500: No trained model

### POST /api/batch_predict
Predict fraud for multiple SMS messages.

**Request Body:**
```json
{
  "texts": ["Message 1", "Message 2", "Message 3"]
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Message 1",
      "fraud_probability": 0.95,
      "is_fraud": true
    },
    {
      "text": "Message 2",
      "fraud_probability": 0.1,
      "is_fraud": false
    }
  ],
  "threshold": 0.5
}
```

### POST /api/chat
Get a response from the chatbot.

**Request Body:**
```json
{
  "question": "What is SMS fraud?",
  "backend": "local",
  "api_key": "optional-api-key",
  "local_model": "optional-model-name"
}
```

**Response:**
```json
{
  "response": "SMS fraud refers to fraudulent activities conducted via SMS messages..."
}
```

**Notes:**
- `backend` can be "local" or "openai"
- `api_key` is required for OpenAI backend
- `local_model` specifies which local model to use

## Deployment

The API can be deployed using:
- Docker (use the provided Dockerfile)
- Heroku (use the Procfile)
- Render, Railway, or other cloud platforms

## Testing Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Train a model: `python train.py --model_type sklearn`
3. Run the server: `python web_app.py`
4. Test endpoints using curl or Postman

Example curl command:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Free money! Click here"}'
```