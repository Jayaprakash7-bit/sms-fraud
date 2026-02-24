# 🚀 Free Public API Deployment Guide

## Step 1: Install Git
1. Download from: https://git-scm.com/download/win
2. Install with default settings
3. Restart VS Code

## Step 2: Set up Repository
Run the `setup_git.bat` file I created, or manually:
```cmd
git init
git add .
git commit -m "Initial commit - SMS Fraud Detection API"
```

## Step 3: Create GitHub Repository
1. Go to https://github.com/Jayaprakash7-bit
2. Click "New repository"
3. Name: `sms-fraud-api`
4. Description: "SMS Fraud Detection API with ML models"
5. Make it **Public**
6. Don't initialize with README
7. Click "Create repository"

## Step 4: Push to GitHub
```cmd
git remote add origin https://github.com/Jayaprakash7-bit/sms-fraud-api.git
git push -u origin main
```

## Step 5: Deploy to Render (Free)
1. Go to https://render.com
2. Sign up/Login (free)
3. Click "New" → "Web Service"
4. Connect GitHub: `Jayaprakash7-bit/sms-fraud-api`
5. Configure:
   - **Name**: sms-fraud-api
   - **Runtime**: Docker
   - **Build Command**: (leave empty)
   - **Start Command**: (leave empty)
6. Click "Create Web Service"

## Step 6: Get Your Public API URL
After 5-10 minutes, you'll get: `https://sms-fraud-api.onrender.com`

## 🧪 Test Your Public API
```bash
curl -X POST https://sms-fraud-api.onrender.com/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Free money! Click here\"}"
```

## 📋 API Endpoints (Public)
- `POST /api/predict` - Single SMS prediction
- `POST /api/batch_predict` - Multiple SMS predictions
- `POST /api/chat` - AI chatbot

## 💡 Alternative Free Options
- **Railway**: https://railway.app (similar to Render)
- **Heroku**: https://heroku.com (750 free hours/month)
- **Fly.io**: https://fly.io (free tier available)

Your API will be publicly accessible once deployed! 🎉