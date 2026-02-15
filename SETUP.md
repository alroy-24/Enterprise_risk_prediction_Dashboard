# Enterprise Risk Intelligence Platform - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional, for contributors)
pip install -r requirements-dev.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Set database credentials, API keys, etc.
```

### 3. Run the Application

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`

---

## 🛠️ Development Setup

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Code Quality Tools

```bash
# Format code with black
black src/ tests/ --line-length=100

# Sort imports
isort src/ tests/ --profile=black

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Security scanning
bandit -r src/ -ll
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models_smoke.py -v

# Run with markers
pytest -m "not slow"
```

---

## 📊 MLflow Tracking

MLflow is enabled by default for experiment tracking.

### Start MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
```

Access the UI at `http://localhost:5000`

### Configuration

Edit [config/model_config.yaml](config/model_config.yaml):

```yaml
model:
  use_mlflow: true

governance:
  mlflow_tracking_uri: "mlruns"  # or http://mlflow-server:5000
  mlflow_experiment_name: "risk_prediction"
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t risk-prediction:latest .
```

### Run Container

```bash
docker run -p 8501:8501 \
  -e POSTGRES_URI=your_db_uri \
  -e MLFLOW_TRACKING_URI=your_mlflow_uri \
  risk-prediction:latest
```

### Docker Compose (Optional)

Create `docker-compose.yml` for full stack deployment with PostgreSQL and MLflow.

---

## 🔧 Configuration

### Environment Variables

Key environment variables (see [.env.example](.env.example)):

- `APP_ENV`: Environment (development/production)
- `POSTGRES_URI`: Database connection string
- `MLFLOW_TRACKING_URI`: MLflow tracking server
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

### YAML Configuration

- [config/model_config.yaml](config/model_config.yaml): Model hyperparameters
- [config/scenarios.yaml](config/scenarios.yaml): Stress test scenarios
- [config/weights.yaml](config/weights.yaml): Risk scoring weights

---

## 📈 Monitoring & Logging

### Structured Logging

Logs are written to:
- Console: Colored, human-readable
- `logs/app.log`: Standard text format with rotation
- `logs/app_json.log`: JSON format for log aggregation

### Log Levels

Set via environment variable:

```bash
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## 🧪 Testing Strategy

### Test Structure

```
tests/
├── test_features.py       # Feature engineering tests
├── test_models_smoke.py   # Model training smoke tests
└── test_scenarios.py      # Scenario testing
```

### Writing Tests

```python
import pytest
from src.models import train_models

def test_model_training():
    """Test model trains successfully."""
    # Your test code here
    pass
```

### Coverage Goals

- Target: 80%+ code coverage
- Critical paths: 100% coverage
- Run: `pytest --cov=src --cov-report=term-missing`

---

## 🚢 Deployment

### Railway

1. Connect GitHub repository
2. Railway auto-detects Dockerfile
3. Set environment variables in Railway dashboard
4. Deploy

### Streamlit Community Cloud

1. Connect GitHub repository
2. Set main file: `src/app.py`
3. Add secrets in dashboard
4. Deploy

### AWS/Azure/GCP

Use the Dockerfile for container-based deployment on:
- AWS ECS/Fargate
- Azure Container Instances
- Google Cloud Run

---

## 📝 Contributing

### Workflow

1. Create feature branch
2. Make changes
3. Run tests and linters
4. Submit pull request
5. CI/CD pipeline runs automatically

### Code Style

- Python: PEP 8 with Black formatting
- Line length: 100 characters
- Type hints: Encouraged
- Docstrings: Google style

---

## 🔐 Security

### Best Practices

- Never commit `.env` files
- Use environment variables for secrets
- Run `bandit` security scanner
- Keep dependencies updated
- Use pre-commit hooks

### Updating Dependencies

```bash
# Check for updates
pip list --outdated

# Update specific package
pip install --upgrade package-name

# Update requirements
pip freeze > requirements.txt
```

---

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 🐛 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt
```

**MLflow UI not starting:**
```bash
# Check if port 5000 is available
# Specify different port
mlflow ui --port 5001
```

**Database connection errors:**
```bash
# Verify POSTGRES_URI in .env
# Check database is running
# Test connection with psql
```

---

## 📞 Support

For issues and questions:
- Check existing GitHub Issues
- Review documentation
- Create new issue with details

---

**Happy Coding! 🚀**
