"""Ensamble por log-pooling: p ∝ p_elo^w * p_dc^(1-w)."""
import numpy as np


def log_pool(p_elo: np.ndarray, p_dc: np.ndarray, w: float) -> np.ndarray:
    p = np.clip(p_elo, 1e-12, 1) ** w * np.clip(p_dc, 1e-12, 1) ** (1 - w)
    return p / p.sum(axis=-1, keepdims=True)


def blend_market(p_model: np.ndarray, p_market, weight: float) -> np.ndarray:
    """Mezcla geométrica (log-pool) del pronóstico del modelo con la probabilidad
    implícita del mercado. `weight` es el peso del mercado en [0, 1]; 0 = solo modelo.
    Devuelve el modelo sin tocar si no hay cuota o el peso es 0."""
    if p_market is None or weight <= 0:
        return p_model
    pm = np.clip(np.asarray(p_market, dtype=float), 1e-12, 1)
    p = np.clip(p_model, 1e-12, 1) ** (1 - weight) * pm ** weight
    return p / p.sum()


def condition_matrix(matrix: np.ndarray, target_1x2: np.ndarray) -> np.ndarray:
    """Reescala la matriz de marcadores DC para que sus marginales 1X2
    coincidan con las probabilidades del ensamble (mantiene la forma DC
    dentro de cada resultado)."""
    out = matrix.copy()
    home = np.tril(np.ones_like(matrix), -1).astype(bool)
    draw = np.eye(matrix.shape[0], dtype=bool)
    away = np.triu(np.ones_like(matrix), 1).astype(bool)
    for mask, target in zip((home, draw, away), target_1x2):
        s = out[mask].sum()
        if s > 0:
            out[mask] *= target / s
    return out / out.sum()
