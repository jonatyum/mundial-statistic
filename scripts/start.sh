#!/usr/bin/env bash
# Mundial 2026 — levanta todo en un comando:
#   actualiza datos -> reentrena -> re-simula -> API + dashboard
#
# Uso:
#   scripts/start.sh                # pipeline con datos frescos + servicios
#   scripts/start.sh --skip-update  # solo levanta API y dashboard
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "✗ Falta el entorno: corré  uv venv && uv pip install -e '.[dev]'"
  exit 1
fi
mkdir -p logs

if [ "${1:-}" != "--skip-update" ]; then
  echo "[1/3] Actualizando datos, modelos y predicciones (puede tardar ~1 min)..."
  "$PY" scripts/run_pipeline.py --fresh
else
  echo "[1/3] (actualización omitida)"
fi

echo "[2/3] Levantando API en http://localhost:8026 (Swagger en /docs)..."
pkill -f "uvicorn mundial.serve.api" 2>/dev/null || true
sleep 1
nohup .venv/bin/uvicorn mundial.serve.api:app --port 8026 >> logs/api.log 2>&1 &

echo "[3/3] Levantando dashboard en http://localhost:8501..."
pkill -f "streamlit run" 2>/dev/null || true
sleep 1
nohup .venv/bin/streamlit run src/mundial/serve/dashboard.py --server.port 8501 \
  >> logs/dashboard.log 2>&1 &

sleep 6
ok=1
curl -s -m 5 http://127.0.0.1:8026/health > /dev/null \
  && echo "  ✓ API respondiendo" || { echo "  ✗ API no responde (logs/api.log)"; ok=0; }
curl -s -m 5 http://127.0.0.1:8501/_stcore/health > /dev/null \
  && echo "  ✓ Dashboard respondiendo" || { echo "  ✗ Dashboard no responde (logs/dashboard.log)"; ok=0; }

if [ "$ok" = 1 ]; then
  open http://localhost:8501 2>/dev/null || true
  echo ""
  echo "Listo ⚽  Dashboard: http://localhost:8501 · API: http://localhost:8026/docs"
  echo "Detener todo: scripts/stop.sh"
fi
