# Plan de Arquitectura — Predicción de Resultados del Mundial 2026

> **Objetivo:** predecir ganadores y marcadores de los partidos de la Copa Mundial FIFA 2026
> (48 selecciones, 104 partidos, 11 jun – 19 jul 2026, sedes en EE. UU., México y Canadá)
> mediante modelos estadísticos y de machine learning, con actualización partido a partido
> durante el torneo.

---

## 1. Visión general del sistema

```
┌─────────────────┐   ┌──────────────────┐   ┌───────────────────┐
│  INGESTA DATOS  │──▶│  FEATURE STORE   │──▶│  ENTRENAMIENTO    │
│  (APIs, scraping│   │  (features por   │   │  (Poisson, Elo,   │
│   CSVs hist.)   │   │   equipo/partido)│   │   GBM, bayesiano) │
└─────────────────┘   └──────────────────┘   └─────────┬─────────┘
                                                       │ modelos serializados
                      ┌──────────────────┐   ┌─────────▼─────────┐
                      │  DASHBOARD / API │◀──│  INFERENCIA +     │
                      │  (predicciones,  │   │  SIMULACIÓN       │
                      │   probabilidades)│   │  MONTE CARLO      │
                      └──────────────────┘   └───────────────────┘
```

Cinco capas desacopladas:

1. **Ingesta** — descarga y normaliza datos históricos y en vivo.
2. **Feature store** — calcula y versiona variables por equipo y por partido.
3. **Entrenamiento** — ajusta varios modelos independientes y un ensamble.
4. **Inferencia + simulación** — probabilidades 1X2, marcador exacto y simulación
   Monte Carlo del torneo completo (avance de fase, campeón).
5. **Presentación** — CLI / API REST / dashboard con las predicciones.

---

## 2. Fuentes de datos

| Fuente | Contenido | Acceso |
|---|---|---|
| Kaggle "International football results 1872–2026" | ~47k partidos internacionales (marcador, sede, torneo) | CSV gratuito |
| FIFA / Elo Ratings (eloratings.net) | Ranking Elo histórico por selección | scraping/CSV |
| Ranking FIFA oficial | Puntos SUM (desde 2018) | scraping mensual |
| Transfermarkt | Valor de mercado de plantillas, edad, club | scraping |
| FBref / StatsBomb open data | xG, posesión, tiros, stats avanzadas (equipo y **jugador**) | CSV/scraping |
| FotMob / SofaScore | Ratings por jugador y por partido, minutos, alineaciones | scraping/API |
| football-data.org o API-Football | Fixtures, alineaciones y resultados en vivo del Mundial | API REST (key) |
| Casas de apuestas (odds agregadas) | Probabilidades implícitas de mercado | API/scraping |
| Datos de contexto | Distancias entre sedes, altitud, clima, huso horario | estático |

**Regla:** todo dato crudo se guarda inmutable en `data/raw/`; las transformaciones
viven en código, nunca a mano.

---

## 3. Variables y estimadores (feature engineering)

### 3.1 Fuerza del equipo
- **Elo rating** (y delta de Elo últimos 6/12 meses — tendencia).
- **Ranking FIFA** (puntos y posición).
- **Ataque/defensa estimados** por modelo Poisson jerárquico (goles esperados a favor
  y en contra vs. rival promedio).
- **Valor de mercado total y mediano de la plantilla** (Transfermarkt, log-escala).
- **Calidad del once inicial**: suma de valor de los 11 titulares cuando hay alineación.

### 3.2 Forma reciente
- Puntos por partido en últimos 5/10/15 partidos (ponderación exponencial).
- Goles a favor / en contra por partido (medias móviles).
- xG a favor y en contra (cuando exista), diferencia xG − goles (sobre/infrarrendimiento).
- Racha (victorias/derrotas consecutivas).
- Rendimiento específico en torneos oficiales vs. amistosos.

### 3.3 Enfrentamiento y contexto del partido
- Historial head-to-head (resultado medio, ponderado por antigüedad).
- **Localía efectiva**: anfitrión (USA/MEX/CAN), distancia de viaje desde el partido
  anterior, huso horario, altitud de la sede (p. ej. Ciudad de México 2240 m), clima.
- Días de descanso entre partidos; minutos acumulados de jugadores clave.
- Fase del torneo (grupos vs. eliminación directa — cambia la dinámica de goles).
- Importancia del partido (¿ya clasificado? ¿se juega la vida?).

### 3.4 Plantilla y jugadores
- Edad media, experiencia internacional media (caps).
- Nº de jugadores en ligas top-5 / en Champions League.
- Bajas y lesiones de titulares (impacto = valor del jugador / valor del once).
- Goleadores: tasa de goles por 90' de los delanteros.

### 3.4b Rendimiento individual reciente (por jugador)
Cada jugador convocado tiene su propio vector de features, calculado sobre sus
**últimos 5/10 partidos** (club + selección, ponderación exponencial por fecha):

- **Producción ofensiva:** goles, asistencias, xG y xA por 90', tiros y pases clave.
- **Rating por partido** (FotMob/SofaScore): media móvil y tendencia (¿llega en alza
  o en baja?).
- **Forma de cierre de temporada de club:** rendimiento en los últimos 2 meses antes
  del Mundial (la temporada europea termina en mayo → señal muy fresca).
- **Carga y fatiga:** minutos acumulados en los últimos 30/60 días, partidos cada
  ≤4 días, ¿viene de jugar prórroga/finales? Riesgo de fatiga al llegar al torneo.
- **Disponibilidad:** lesiones recientes, minutos desde la vuelta de una lesión,
  riesgo de suspensión (amarillas acumuladas durante el torneo).
- **Porteros:** goles evitados (PSxG − goles recibidos), % paradas, rendimiento en penales.
- **Defensas:** duelos ganados, intercepciones, errores que terminan en gol.

**Agregación jugador → equipo** (lo que consume el modelo de partido):
- Suma/media ponderada por minutos esperados del **once probable** (si hay alineación
  confirmada, se usa la real).
- "Fuerza ofensiva del once" = Σ (xG+xA por 90' de titulares); ídem defensiva.
- Índice de forma del equipo = media de ratings recientes de los 11 titulares.
- Índice de fatiga del equipo = media de carga de minutos de los titulares.
- Delta vs. plantilla completa: cuánto pierde el equipo si faltan los lesionados.

Durante el torneo, estas features se **recalculan tras cada jornada** con los
ratings y minutos de los propios partidos del Mundial.

### 3.5 Variables de mercado (benchmark y feature)
- Probabilidades implícitas de las casas de apuestas (quitando el margen).
  Se usan tanto como **feature** opcional como **baseline a batir**.

### 3.6 Confederación y estilo
- Confederación (UEFA, CONMEBOL, etc.) y rendimiento histórico inter-confederación.
- Ritmo de juego (tiros/90, posesión) como proxies de estilo.

### 3.7 Variables adicionales — prioridad alta (incluir desde el inicio)
Salen de fuentes que ya se ingieren (FBref, historial de partidos, Elo):

- **Balón parado:** % de goles a favor/en contra de córner, tiro libre y penal;
  presencia de especialista. En torneos cortos los set pieces deciden muchos
  partidos (~30 % de los goles en Qatar 2022).
- **Entrenador:** tiempo en el cargo, win rate con la selección, experiencia previa
  en mundiales, cambio reciente de DT ("new manager bounce", efecto que se desvanece).
- **Cohesión del equipo:** minutos que los titulares han jugado *juntos*, % de la
  plantilla que repite de las eliminatorias, parejas consolidadas
  (central-central, doble pivote).
- **Strength of schedule:** toda la forma reciente (3.2) se ajusta por el Elo de los
  rivales enfrentados — ganar amistosos débiles ≠ ganar en eliminatorias CONMEBOL.
- **Disciplina:** tarjetas y faltas por partido → riesgo de expulsión y de perder
  titulares por suspensión en rondas KO.

### 3.8 Variables adicionales — prioridad media (Fases 3–4 del roadmap)
- **Experiencia en eliminación directa:** minutos de la plantilla en fases KO de
  mundiales/copas continentales/Champions; historial del país en tandas de penales.
- **Calor y horario:** índice de calor por sede y hora (Dallas, Houston, Miami,
  Monterrey en jun-jul; partidos al mediodía por TV europea). Penaliza más a
  equipos de ligas frías; se cruza con la feature de fatiga (3.4b).
- **Localía de afición:** México juega de local real, pero las diásporas en EE. UU.
  convierten en "local" a varias selecciones según la sede (aprox. con datos
  demográficos por ciudad).
- **Consistencia/volatilidad:** varianza del rendimiento, no solo la media. Un
  equipo volátil sorprende más en KO y cae más en grupos; entra directamente en
  la simulación Monte Carlo.

### 3.9 Variables opcionales — valor incierto (solo si sobra tiempo)
- **Matchup de estilos:** interacciones presión alta vs. salida de balón, bloque
  bajo vs. remate exterior. Requiere datos de eventos; riesgo alto de sobreajuste.
- **Movimiento de líneas de apuestas** (line movement): incorpora noticias de
  alineación antes que cualquier scraper.
- **Árbitro asignado:** tendencia a tarjetas/penales; señal pequeña, se conoce ~48 h antes.

> ⚠️ **Control de sobreajuste:** cada variable nueva añade riesgo con tan pocos
> partidos de torneo para validar. El backtest contra 2018/2022 (sección 6) decide
> qué features sobreviven; las que no mejoran log-loss/RPS se descartan.

---

## 4. Modelos estadísticos

Se entrenan **varios modelos independientes** y luego se combinan:

### 4.1 Poisson bivariado / Dixon-Coles (núcleo clásico)
- Goles de cada equipo ~ Poisson(λ), con λ = f(ataque_i, defensa_j, localía).
- Corrección Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1).
- Ponderación temporal exponencial (ξ ≈ 0.0019/día) para dar más peso a lo reciente.
- **Salida:** matriz de probabilidades de marcador exacto → 1X2, over/under, etc.

### 4.2 Modelo bayesiano jerárquico (PyMC / Stan)
- Ataque y defensa como efectos aleatorios por selección, con priors por
  confederación y pooling parcial (clave para selecciones con pocos partidos).
- Skellam para la diferencia de goles.
- **Ventaja:** incertidumbre completa (distribuciones posteriores, no point estimates).

### 4.3 Elo probabilístico
- P(victoria) = 1 / (1 + 10^(−ΔElo/400)), ajustado por localía y empates
  (extensión Davidson para el empate).
- Sirve de modelo simple, robusto y de control.

### 4.4 Gradient Boosting (XGBoost / LightGBM)
- Clasificador multiclase (victoria/empate/derrota) y regresores de goles,
  usando TODAS las features de la sección 3.
- Calibración posterior de probabilidades (isotónica o Platt).
- Validación temporal estricta (entrenar pasado → validar futuro; nunca al revés).

### 4.5 Modelo de orden (ordered logit / Bradley-Terry extendido)
- Resultado ordinal (derrota < empate < victoria) en función de la diferencia
  de fuerza latente. Complementario y muy interpretable.

### 4.6 Ensamble
- Combinación de las probabilidades de 4.1–4.5 por **log-pooling con pesos
  optimizados** sobre validación (minimizando log-loss / RPS).
- Las cuotas de mercado pueden entrar como un "modelo" más del pool.

---

## 5. Simulación Monte Carlo del torneo

1. Para cada partido pendiente, el ensamble produce P(marcadores).
2. Se simula el torneo completo **N = 100 000 veces**: fase de grupos (formato 2026:
   12 grupos de 4, avanzan 1º, 2º y los 8 mejores terceros), dieciseisavos,
   octavos… hasta la final, respetando el cuadro real de cruces.
3. **Desempates de grupo según reglamento FIFA:** puntos → diferencia de gol →
   goles a favor → head-to-head → fair play (tarjetas) → sorteo. Con 12 grupos y
   la pelea por los 8 mejores terceros, los empates de puntos son frecuentes;
   simularlos mal distorsiona las P(clasificar). Mismo orden de criterios para
   el ranking de terceros.
4. En eliminatorias, la **prórroga se simula con su propia tasa de gol**
   (λ reducida: equipos cansados y conservadores anotan a menor ritmo que en los
   90'); usar la λ del partido regular sobreestima goles y subestima tandas.
   Los penales ≈ 50/50 con leve ajuste por experiencia/ranking.
5. **Salidas:** P(pasar de grupo), P(llegar a cada ronda), P(campeón) por selección,
   distribución de goles del torneo, partidos más parejos, etc.
6. Tras cada jornada real se **reentrena/actualiza** (Elo y forma cambian) y se
   re-simula → las probabilidades evolucionan durante el torneo.

---

## 6. Evaluación y backtesting

- **Métricas:** log-loss, Brier score y **Ranked Probability Score (RPS)** para 1X2;
  log-loss de marcador exacto; calibración (reliability diagrams).
- **Backtesting:** entrenar con datos hasta 2017 → evaluar en Mundial 2018;
  hasta 2021 → Mundial 2022; hasta 2023 → Eurocopa/Copa América 2024. Eso da una
  estimación honesta del rendimiento esperado en 2026.
- **Baselines a batir:** (a) probabilidades implícitas de las casas de apuestas,
  (b) Elo puro, (c) predicción uniforme.
- Tests de no-fuga temporal: ninguna feature puede usar información posterior
  a la fecha del partido (asserts automáticos en el feature store).

---

## 7. Estructura del proyecto

```
mundial-stadistic/
├── pyproject.toml              # deps: pandas, numpy, scipy, scikit-learn,
│                               # xgboost, lightgbm, pymc, arviz, statsmodels,
│                               # requests, beautifulsoup4, fastapi, streamlit
├── config/
│   ├── settings.yaml           # rutas, API keys (via env), N simulaciones
│   └── tournament_2026.yaml    # grupos, calendario, sedes, formato de cruces
├── data/
│   ├── raw/                    # datos crudos inmutables (gitignored)
│   ├── interim/                # limpiados/normalizados
│   ├── processed/              # features listas para entrenar
│   ├── predictions/            # prediction log append-only (sección 9.4)
│   └── snapshots/              # feature store fechado por jornada (sección 9.5)
├── src/mundial/
│   ├── ingest/                 # un módulo por fuente (kaggle.py, elo.py,
│   │                           #   transfermarkt.py, fbref.py, fotmob.py,
│   │                           #   live_api.py, odds.py)
│   ├── features/               # builders: strength.py, form.py, h2h.py,
│   │                           #   context.py, squad.py, player.py (rendimiento
│   │                           #   individual + agregación al once),
│   │                           #   set_pieces.py, coach.py, cohesion.py,
│   │                           #   discipline.py  (+ feature store en parquet)
│   ├── models/
│   │   ├── poisson_dc.py       # Dixon-Coles
│   │   ├── bayes_hier.py       # jerárquico PyMC
│   │   ├── elo_model.py
│   │   ├── gbm.py              # XGBoost/LightGBM + calibración
│   │   ├── ordered.py          # ordered logit
│   │   └── ensemble.py         # log-pooling con pesos optimizados
│   ├── simulate/
│   │   ├── match.py            # P(marcador) → resultado simulado
│   │   └── tournament.py       # Monte Carlo del cuadro completo 2026
│   ├── evaluate/               # métricas, calibración, backtests
│   └── serve/
│       ├── api.py              # FastAPI: /predict, /odds, /champion-probs
│       └── dashboard.py        # Streamlit: 7 vistas (ver sección 12)
├── notebooks/                  # exploración (nunca lógica de producción)
├── tests/                      # unit + tests de no-fuga temporal
└── scripts/
    ├── update_daily.py         # ingesta + features + reentrenar + re-simular
    ├── live_poll.py            # poller continuo en días de partido (resultados,
    │                           #   alineaciones, cuotas) — ver sección 9
    └── backtest.py
```

**Stack:** Python 3.12, pandas/polars, scikit-learn, XGBoost/LightGBM, PyMC,
statsmodels, FastAPI + Streamlit, parquet como formato de intercambio,
`uv` para gestión de dependencias. Todo reproducible con seeds fijos.

---

## 8. Roadmap de implementación

| Fase | Entregable | Duración est. |
|---|---|---|
| 0. Setup | git init, `pyproject.toml` (uv), API keys, tabla canónica de equipos | horas |
| 1. Datos | Ingesta histórica (Kaggle + Elo + rankings) y limpieza | 1–2 días |
| 1b. Evaluación | Harness de backtest + métricas ANTES que los modelos | 0.5–1 día |
| 2. Baseline | Elo + Poisson Dixon-Coles con backtest 2018/2022 | 1–2 días |
| 3. Features | Feature store completo (secciones 3.1–3.6) | 2–3 días |
| 4. ML | GBM calibrado + bayesiano jerárquico + ensamble | 3–4 días |
| 5. Simulación | Monte Carlo del formato 2026 (grupos + 8 mejores terceros) | 1–2 días |
| 6. Servir | API + dashboard + script de actualización diaria | 1–2 días |
| 7. Torneo | Ciclo en vivo: tras cada jornada → actualizar → re-simular | continuo |

> ⚠️ El torneo **empieza hoy (11-jun-2026)**: la prioridad es llegar rápido a la
> Fase 2 (baseline funcional con Elo + Dixon-Coles) para tener predicciones desde ya,
> y mejorar el modelo mientras avanza la fase de grupos.

---

## 9. Actualización continua — datos siempre frescos

**Principio:** ninguna request se responde con datos obsoletos. Se combinan dos
mecanismos: actualización programada en segundo plano y verificación de frescura
en el momento de cada request.

### 9.1 Frecuencias de actualización por fuente

| Dato | Frecuencia | Mecanismo |
|---|---|---|
| Resultados en vivo (API-Football) | cada 1–5 min en días de partido | poller `live_poll.py` |
| Alineaciones confirmadas | ~1 h antes de cada partido | poller |
| Cuotas de apuestas | cada 15–30 min en días de partido; 6 h resto | poller |
| Lesiones / bajas / suspensiones | diaria | `update_daily.py` |
| Ratings de jugadores (FotMob/SofaScore) | tras cada jornada | `update_daily.py` |
| Elo, forma, features de equipo | tras cada jornada (recálculo completo) | `update_daily.py` |
| Reentrenamiento de modelos + re-simulación Monte Carlo | tras cada jornada | `update_daily.py` |
| Valores de mercado (Transfermarkt) | semanal | `update_daily.py` |
| Ranking FIFA | mensual (cuando FIFA publica) | `update_daily.py` |

### 9.2 Mecanismos

1. **Scheduler en segundo plano:** `scripts/update_daily.py` programado vía
   `launchd` (macOS) o cron — corre la cadena completa ingesta → features →
   reentrenar → re-simular. En días de partido, `scripts/live_poll.py` corre como
   proceso continuo consultando la API en vivo.
2. **Freshness-on-read (garantía en cada request):** cada tabla del feature store
   guarda su `last_updated`. Cuando llega una request a la API:
   - Si el dato está dentro de su TTL (tabla 9.1) → responde directo.
   - Si está vencido → dispara refresco de esa fuente *antes* de responder
     (con timeout; si la fuente externa falla, responde con el último dato
     disponible y lo marca).
   - **Toda respuesta incluye metadatos de frescura:** `last_updated` por fuente
     y `model_version`, para que siempre se sepa con qué información se predijo.
3. **Invalidación por evento:** un resultado final nuevo invalida automáticamente
   features de forma, Elo y simulaciones del torneo → se encola el recálculo
   sin esperar al cron diario.
4. **Detección de datos rotos:** si un scraper devuelve vacío o valores anómalos
   (validación con pandera/great-expectations), se conserva el último dato bueno
   y se registra alerta — nunca se sobreescribe un dato válido con basura.

### 9.3 Reglas de predicción "siempre actual"

- `/predict` nunca lee modelos serializados viejos: usa siempre el último
  `model_version` y el feature store vigente.
- Si hay alineación confirmada, la predicción se recalcula automáticamente con
  el once real (sección 3.4b) y reemplaza a la basada en el once probable.
- Las probabilidades de campeón publicadas siempre corresponden a la última
  simulación posterior al último partido terminado.

### 9.4 Registro inmutable de predicciones (prediction log)

Al pitazo inicial de cada partido se **congela** la predicción oficial en
`data/predictions/` (append-only, nunca se sobreescribe): probabilidades 1X2,
matriz de marcadores, marcador esperado, `model_version`, features usadas y
timestamp. Es el track record del proyecto:

- Permite demostrar al final del torneo qué tan bien predijo el modelo de verdad
  (las predicciones "vivas" se sobreescriben con cada actualización; sin el log
  no hay evidencia).
- Habilita la comparación honesta contra el mercado **durante** el torneo, no
  solo en backtest.
- Alimenta el monitoreo de calibración en vivo (sección 12).

### 9.5 Snapshots del feature store por jornada

Tras cada actualización se guarda copia fechada del feature store
(`data/snapshots/YYYY-MM-DD/*.parquet`). Permite responder "¿qué sabía el modelo
el 15 de junio?" y reproducir cualquier predicción pasada. Coste: MB por día.

---

## 10. Riesgos y mitigaciones

- **Pocas observaciones por selección** → pooling parcial bayesiano y priors por confederación.
- **Selecciones debutantes** (formato de 48 trae equipos nuevos) → respaldarse en valor
  de plantilla y Elo, que existen aunque no haya historial mundialista.
- **Fuga temporal** → validación estrictamente cronológica + tests automáticos.
- **Scraping frágil** (Transfermarkt/FBref) → cachear crudos, degradar con elegancia
  si una fuente falla (el ensamble funciona con subconjuntos de features).
- **Sobreajuste al ruido del fútbol** → el fútbol tiene techo de predictibilidad
  (~55-60 % acierto 1X2 incluso para el mercado); medir contra baselines, no contra 100 %.

---

## 11. Checklist previo a la implementación

Recomendaciones de orden y trampas conocidas, en la secuencia exacta a seguir:

### Fase 0 — Setup (hoy, antes de escribir lógica)
- [ ] `git init` + primer commit con este documento. `.gitignore` excluye `data/raw/`,
      `data/interim/`, `.env` y modelos serializados.
- [ ] `pyproject.toml` gestionado con `uv`; versiones pineadas; seeds fijos en config.
- [ ] **Registrar API keys hoy mismo** (API-Football / football-data.org): los tiers
      gratuitos tienen límites de requests y la activación puede demorar. Es el único
      paso con reloj externo — el torneo ya empezó.
- [ ] **Tabla canónica de IDs de selección** (`config/teams.yaml`): cada fuente nombra
      distinto ("USA"/"United States", "Korea Republic"/"South Korea",
      "Côte d'Ivoire"/"Ivory Coast"). Toda ingesta mapea al ID canónico al entrar;
      un join que falla por nombre falla en silencio — es la trampa nº 1 de estos
      proyectos.

### Orden de construcción (anti-trampas)
- [ ] **Evaluación antes que modelos:** implementar `evaluate/` (log-loss, RPS, Brier)
      y el harness de backtest 2018/2022 con baselines uniforme y Elo puro, ANTES
      del primer modelo. Cada modelo nuevo debe demostrar que aporta desde su primer
      commit; al revés se termina con cinco modelos y ninguna comparación honesta.
- [ ] **Empezar por fuentes estables, no por scrapers:** CSV de Kaggle (descarga
      directa) + Elo ratings bastan para el baseline. Transfermarkt y FBref bloquean
      y cambian HTML → Fase 3. Para valores de mercado existen dumps en Kaggle que
      evitan scrapear.
- [ ] **Verificar el formato 2026 contra el reglamento oficial FIFA:** la asignación
      de llaves de los 8 mejores terceros depende de QUÉ grupos aportan los terceros
      (no es trivial). Codificarlo mal invalida toda la simulación Monte Carlo →
      tests unitarios con casos del reglamento en `tests/test_tournament.py`.

### Principios permanentes
- [ ] **Expectativas calibradas:** el objetivo realista es acercarse al mercado en
      log-loss, no superarlo. Si el backtest "le gana" al mercado por mucho margen,
      casi seguro hay fuga temporal — investigar antes de celebrar.
- [ ] **Infra mínima:** parquet + DuckDB para consultas; nada de Postgres ni
      servicios. ~50k filas y una persona — la complejidad de infra roba días que
      no hay con el torneo en marcha.
- [ ] **Todo timestamp a UTC al ingerir** y conversión solo al mostrar. Tres países
      anfitriones, 4+ husos horarios y fuentes que reportan en hora local — el bug
      clásico es calcular mal los "días de descanso" por mezclar zonas.
- [ ] **Scraping responsable:** rate limiting, caché agresiva de crudos y user-agent
      identificable. Un baneo de Transfermarkt/FBref a mitad de torneo deja al
      sistema ciego justo cuando más datos necesita.

**Secuencia de arranque:** git init + deps → tabla canónica de equipos → ingesta
Kaggle/Elo → harness de evaluación → baseline Elo + Dixon-Coles. Con eso hay
predicciones honestas la primera semana, con la fase de grupos aún en juego.

---

## 12. Parte visual — Dashboard

**Stack:** Streamlit + Plotly (interactivo, cero frontend custom). Corre local con
`streamlit run src/mundial/serve/dashboard.py` y lee del feature store y la API.
Toda vista muestra `last_updated` y `model_version` (sección 9.2).

### 12.1 Vistas principales

1. **Próximos partidos** — tarjeta por partido: probabilidades 1X2 (barras),
   marcadores más probables, goles esperados por equipo, y comparación
   lado a lado con la probabilidad implícita del mercado (¿dónde discrepa el modelo?).
2. **Ficha de partido** — matriz de marcadores exactos como heatmap, y los
   factores que más pesan en la predicción (importancias SHAP del GBM:
   "Elo +120, fatiga del rival, baja de su delantero titular…"). Hace la
   predicción explicable, no una caja negra.
3. **Cuadro del torneo (bracket)** — el cuadro completo con P(llegar a cada ronda)
   por selección; se actualiza tras cada simulación Monte Carlo.
4. **Tablas de grupos simuladas** — por grupo: P(1º), P(2º), P(mejor tercero),
   P(eliminado) y la tabla esperada de puntos.
5. **Carrera al título** — evolución temporal de P(campeón) de las principales
   selecciones (línea por equipo, un punto por jornada) usando el prediction log;
   es la vista más narrativa del torneo.
6. **Ficha de selección** — Elo histórico, forma reciente, once probable con
   ratings individuales (3.4b), lesionados/suspendidos, índice de fatiga.
7. **Salud del modelo** — log-loss/Brier acumulado del modelo vs. mercado durante
   el torneo, reliability diagram en vivo, y el historial de predicciones
   congeladas vs. resultados reales (sección 9.4). Detecta descalibración a tiempo.

### 12.2 Principios

- El dashboard es **solo lectura** del feature store/predictions: nunca dispara
  ingestas ni reentrenos (eso es del scheduler, sección 9) — así no se bloquea.
- Probabilidades siempre con dos decimales y barras, nunca solo "ganará X":
  comunicar incertidumbre es parte del producto.
- Auto-refresh en días de partido (st.rerun con TTL corto) para seguir la jornada.
