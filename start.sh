#!/usr/bin/env bash
# Mundial 2026 — un solo comando: detiene lo previo, actualiza y levanta todo.
#   ./start.sh                # pipeline completo + API + dashboard
#   ./start.sh --skip-update  # solo servicios (rápido)
exec "$(dirname "$0")/scripts/start.sh" "$@"
