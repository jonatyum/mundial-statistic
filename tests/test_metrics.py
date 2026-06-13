import numpy as np

from mundial.evaluate import metrics


def test_log_loss_perfect_and_uniform():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert metrics.log_loss(probs, np.array([0])) < 1e-9
    uni = np.full((10, 3), 1 / 3)
    assert abs(metrics.log_loss(uni, np.zeros(10, dtype=int)) - np.log(3)) < 1e-9


def test_rps_orders_outcomes():
    """RPS castiga menos equivocarse hacia el resultado vecino (empate)."""
    outcome = np.array([0])  # ganó el local
    near = np.array([[0.0, 1.0, 0.0]])   # predijo empate
    far = np.array([[0.0, 0.0, 1.0]])    # predijo visita
    assert metrics.rps(near, outcome) < metrics.rps(far, outcome)


def test_brier_bounds():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert 0 <= metrics.brier(probs, np.array([1])) <= 2


def test_outcome_index():
    assert metrics.outcome_index(2, 1) == 0
    assert metrics.outcome_index(1, 1) == 1
    assert metrics.outcome_index(0, 3) == 2
