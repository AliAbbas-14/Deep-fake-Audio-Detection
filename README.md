# Deepfake Audio Detection & Software Defect Prediction

Professional deepfake audio detection platform for **FAST NUCES AI-2002 Assignment #3**.

## Features

- Urdu, General, and AI Voice Clone detection modules
- Software Defect Prediction (40 features)
- ML models: SVM, Logistic Regression, Perceptron (+ DNN locally)
- Streamlit UI for local development
- Vercel deployment with static frontend + Python serverless API

## Project Structure

```
├── api/                    # Vercel serverless API routes
├── public/                 # Vercel static frontend
├── pages/                  # Streamlit multi-page app
├── utils/                  # Streamlit shared utilities
├── models/                 # Trained sklearn model files
├── scripts/generate_models.py
├── streamlit_app.py        # Local Streamlit home page
└── vercel.json
```

## Local Development (Streamlit)

```bash
pip install -r requirements-streamlit.txt
python models.py
streamlit run streamlit_app.py
```

## Deploy on Vercel

Streamlit cannot run directly on Vercel. This repo includes a Vercel-native version:

- **Frontend:** `public/` (HTML/CSS/JS)
- **Backend:** `api/predict_audio.py`, `api/predict_defect.py`
- **Models:** generated at build time via `scripts/generate_models.py`

### Steps

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **Add New Project**
3. Import `AliAbbas-14/Deep-fake-Audio-Detection`
4. Framework preset: **Other**
5. Deploy

### Deploy via CLI

```bash
npm i -g vercel
vercel login
vercel link
vercel --prod
```

### Vercel Notes

- DNN (PyTorch) is **not** included on Vercel due to serverless size limits
- Vercel supports: **SVM**, **Logistic Regression**, **Perceptron**
- For full DNN support, use local Streamlit or Streamlit Cloud

## Developer

- **Name:** Ali Abbas
- **GitHub:** https://github.com/AliAbbas-14/Deep-fake-Audio-Detection
- **LinkedIn:** https://www.linkedin.com/in/aliabbas1065/
