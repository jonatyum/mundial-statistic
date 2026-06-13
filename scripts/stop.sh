#!/usr/bin/env bash
# Detiene la API y el dashboard del Mundial 2026.
pkill -f "uvicorn mundial.serve.api" 2>/dev/null && echo "✓ API detenida" || echo "API no estaba corriendo"
pkill -f "streamlit run" 2>/dev/null && echo "✓ Dashboard detenido" || echo "Dashboard no estaba corriendo"
