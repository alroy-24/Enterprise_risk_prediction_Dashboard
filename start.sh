#!/bin/bash
# Railway startup script - uses PORT env var if available
PORT=${PORT:-8501}
streamlit run src/app.py --server.address=0.0.0.0 --server.port=$PORT


