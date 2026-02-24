# Deployment Guide

This guide explains how to deploy the SMS Fraud Detection API to various cloud platforms.

## Prerequisites

- A trained model in `models/best/` (run `python train.py --model_type sklearn` if not present)
- Git repository initialized and committed

## Option 1: Deploy to Render (Free Tier Available)

1. Sign up for a free account at [render.com](https://render.com)

2. Connect your GitHub repository:
   - Go to Dashboard > New > Web Service
   - Connect your Git repository
   - Set the following:
     - **Runtime**: Docker
     - **Build Command**: (leave empty, uses Dockerfile)
     - **Start Command**: (leave empty, uses CMD from Dockerfile)

3. Deploy and get your API URL (e.g., `https://your-app-name.onrender.com`)

## Option 2: Deploy to Railway

1. Sign up at [railway.app](https://railway.app)

2. Create a new project from your GitHub repo

3. Railway will automatically detect the Dockerfile and deploy

## Option 3: Deploy to Heroku

1. Install Heroku CLI

2. Login and create app:
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. Deploy:
   ```bash
   git push heroku main
   ```

4. The Procfile will be used to start the app

## Option 4: Local Docker Deployment

```bash
docker build -t sms-fraud-api .
docker run -p 5000:5000 sms-fraud-api
```

## Testing Deployed API

Once deployed, test with:

```bash
curl -X POST https://your-api-url/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Free money! Click here"}'
```

## Environment Variables

For production, consider setting:
- `PORT` (automatically set by most platforms)
- `HF_TOKEN` (for Hugging Face models, if using transformers)
- `OPENAI_API_KEY` (for OpenAI chat backend)

## Notes

- The model files are included in the Docker image
- First request may be slow due to model loading
- Free tiers have usage limits; upgrade for production use