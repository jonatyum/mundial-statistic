"""Métricas de evaluación probabilística. Orden de clases: [local, empate, visita]."""
import numpy as np

EPS = 1e-12


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """probs: (n, 3); outcomes: (n,) con índices 0/1/2."""
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.eye(3)[outcomes]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score: respeta el orden local > empate > visita."""
    onehot = np.eye(3)[outcomes]
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2.0))


def outcome_index(home_score, away_score) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2
