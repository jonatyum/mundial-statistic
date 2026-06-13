"""Ensamble por log-pooling: p ∝ p_elo^w * p_dc^(1-w)."""
import numpy as np


def log_pool(p_elo: np.ndarray, p_dc: np.ndarray, w: float) -> np.ndarray:
    p = np.clip(p_elo, 1e-12, 1) ** w * np.clip(p_dc, 1e-12, 1) ** (1 - w)
    return p / p.sum(axis=-1, keepdims=True)


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
