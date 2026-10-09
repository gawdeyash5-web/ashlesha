# Cosmic Identity — Stellar Object Classification

A complete, production-ready machine learning system and interactive Streamlit dashboard for classifying astronomical survey records into **STAR**, **QSO (Quasars)**, and **GALAXY**.

---

## 🚀 Live Cloud Deployment Options

### Option 1: Streamlit Community Cloud (Recommended — 100% Free & 1-Click)
1. Fork or open the repository on GitHub: [gawdeyash5-web/ashlesha](https://github.com/gawdeyash5-web/ashlesha)
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"** and fill in:
   - **Repository:** `gawdeyash5-web/ashlesha`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** The application will build and be live worldwide with a public URL in ~1 minute.

---

### Option 2: Docker Container Deployment
You can build and run this application anywhere Docker is installed (Local, AWS ECS, GCP Cloud Run, Azure App Service, DigitalOcean, or Render):

#### Using Docker CLI:
```bash
# Build container image
docker build -t cosmic-identity .

# Run container on port 8501
docker run -d -p 8501:8501 --name cosmic-identity-app cosmic-identity
```
Access at `http://localhost:8501`.

#### Using Docker Compose:
```bash
docker compose up -d --build
```

---

### Option 3: Deploy to Render / Koyeb / Railway / Heroku
The repository contains a `Procfile` ready for automatic PaaS deployment:
- Connect your GitHub repository to [Render](https://render.com) as a **Web Service**.
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

---

## 💻 Local Development Setup

### Quick Start (Windows)
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run dashboard locally
streamlit run app.py
```
Or simply double-click **`run_dashboard.bat`**.

To re-train the Random Forest model and regenerate metrics/plots:
```powershell
python main.py
```
Or double-click **`run_model.bat`**.

---

## 📁 Repository Structure
```text
├── .github/workflows/
│   └── ci.yml               # Automated CI test workflow
├── .streamlit/
│   └── config.toml          # Production server & theme config
├── outputs/
│   ├── confusion_matrix.png # Generated confusion matrix plot
│   ├── cosmic_model.joblib  # Serialized trained model
│   ├── feature_importance.png# Top predictive astronomical features
│   ├── metrics.json         # Accuracy, precision, recall, F1 scores
│   └── sample_predictions.csv
├── app.py                   # Streamlit interactive application
├── cosmic_ml.py             # Data cleaning and scikit-learn ML pipeline
├── Dataset 41.csv           # Astronomical survey dataset
├── docker-compose.yml       # Multi-container orchestration
├── Dockerfile               # Production multi-platform container spec
├── main.py                  # Standalone training and evaluation script
├── Procfile                 # Cloud PaaS deployment entrypoint
├── requirements.txt         # Pinned production Python dependencies
└── run_dashboard.bat        # Local Windows dashboard launcher
```

---

## 🔬 Machine Learning Pipeline
1. **Cleaning & Validation:** Normalizes coordinate bounds and astronomical magnitudes (`U`, `G`, `R`, `I`, `Z`) and `Redshift`. Target leakage is prevented by isolating features.
2. **Imputation:** Stratified training median imputer applied strictly to avoid data snooping.
3. **Model:** Balanced Random Forest Classifier with evaluated per-class precision, recall, and F1 scores.
4. **Outputs:** Live interactive inference, dataset explorer, model evaluation metrics, and feature importance visualizer.
