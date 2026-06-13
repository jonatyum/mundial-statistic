# mundial-stadistic

Predicción de ganadores y resultados del Mundial 2026 con modelos estadísticos.
Arquitectura completa en [`ARCHITECTURE.md`](ARCHITECTURE.md); predicciones
actuales en [`REPORT.md`](REPORT.md).

## Estado actual (v0.1)

- ✅ Ingesta del histórico (~49k partidos internacionales 1872–2026, con el
  calendario real del Mundial 2026 incluido).
- ✅ Elo propio (convención eloratings.net: K por torneo, margen de victoria, localía).
- ✅ Modelo Elo-Davidson (empate) + Poisson Dixon-Coles (ponderación temporal,
  shrinkage) + ensamble por log-pooling con peso elegido por backtest.
- ✅ Backtest anti-fuga sobre los Mundiales 2018 y 2022 (log-loss, Brier, RPS)
  contra baselines.
- ✅ Simulación Monte Carlo del formato 2026 completo: 12 grupos, desempates,
  ranking de terceros, matching de terceros al bracket R32, prórroga con λ
  reducida, penales con ajuste Elo. 100k sims en ~2 s.
- ✅ Prediction log append-only (`data/predictions/prediction_log.csv`).
- ✅ Pronóstico partido a partido de los 104 partidos (`REPORT.md` +
  `data/processed/*.csv`): 1X2, goles esperados, marcadores probables,
  over/under, ambos anotan, cruces KO más probables.
- ✅ Actualización automática: LaunchAgent `com.mundial.update` corre el
  pipeline a las 07:00 y 23:30; snapshots fechados en `data/snapshots/`.
- ✅ Calibración en vivo (predicciones congeladas vs. resultados reales).
- ✅ Dashboard Streamlit (5 vistas) + API FastAPI con metadatos de frescura.
- ✅ Benchmark de mercado (requiere `ODDS_API_KEY` de the-odds-api.com).
- ⏳ Pendiente (ver roadmap en ARCHITECTURE.md): features de jugadores, GBM,
  bayesiano jerárquico, alineaciones en vivo.

## Web pública (GitHub Pages)

El sitio estático vive en `docs/index.html` (mismo diseño que el dashboard,
con datos embebidos + JS para filtros, desplegables, gráficos y marcadores en
vivo de ESPN). GitHub Actions lo regenera 3×/día y lo publica.

```bash
.venv/bin/python scripts/build_site.py   # regenera docs/ desde los CSV
```

Activar Pages (una vez): repo → **Settings → Pages → Source: Deploy from a
branch → Branch: main / carpeta /docs → Save**. La URL queda en
`https://<usuario>.github.io/mundial-statistic/`.

## Uso local

**Todo en un comando (desde la raíz del proyecto):**

```bash
./start.sh                # detiene lo previo + actualiza datos + API + dashboard
./start.sh --skip-update  # solo levanta los servicios (rápido)
scripts/stop.sh           # apaga todo
```

Abre solo el navegador en el dashboard (http://localhost:8501);
la API queda en http://localhost:8026/docs. Procesos con `nohup`:
sobreviven al cierre de la terminal.

Comandos individuales:

```bash
uv venv && uv pip install -e ".[dev]"

# pipeline completo: ingesta -> modelos -> pronósticos -> simulación -> REPORT.md
.venv/bin/python scripts/run_pipeline.py --fresh

# vista del día (pronósticos + resultados + evaluación)
.venv/bin/python scripts/today.py

# dashboard (http://localhost:8501)
.venv/bin/streamlit run src/mundial/serve/dashboard.py

# API REST (http://localhost:8026/docs)
.venv/bin/uvicorn mundial.serve.api:app --port 8026

# tests
.venv/bin/python -m pytest -q
```

La actualización automática ya corre 2 veces al día vía launchd
(log en `logs/update.log`). Para desinstalarla:
`launchctl bootout gui/$(id -u)/com.mundial.update && rm ~/Library/LaunchAgents/com.mundial.update.plist`.

Benchmark contra el mercado (opcional): `export ODDS_API_KEY=...` y
`.venv/bin/python -c "from mundial.ingest.odds import compare_with_model; compare_with_model()"`
genera `data/processed/model_vs_market.csv`.

## Resultados de backtest (fase de grupos, 48 partidos c/u)

| Modelo | 2018 log-loss | 2022 log-loss |
|---|---|---|
| Uniforme | 1.099 | 1.099 |
| Elo-Davidson | 0.952 | 1.086 |
| Dixon-Coles | 0.969 | 1.068 |
| **Ensamble (w=0.5)** | **0.953** | **1.065** |

Referencia: las casas de apuestas rondan ~0.95 (2018) y ~1.04 (2022).
