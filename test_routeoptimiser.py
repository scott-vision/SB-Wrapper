import time
import pytest

np = pytest.importorskip("numpy")
from sbs_interface.SBPointOptimiser import RouteOptimizer


def _make_points(n: int = 12):
    rng = np.random.default_rng(0)
    pts = rng.random((n, 2)) * 100.0
    return [(float(x), float(y), 0.0, 0.0) for x, y in pts]


def test_stochastic_not_worse_than_nn():
    pts = _make_points(15)
    opt = RouteOptimizer(pts)
    baseline = opt.optimise()
    np.random.seed(1)
    improved = opt.optimise_stochastic(restarts=5, max_iter=5)
    assert opt._tour_length(improved) <= opt._tour_length(baseline)


def test_stochastic_faster_than_sa():
    pts = _make_points(10)
    opt = RouteOptimizer(pts)
    np.random.seed(1)
    start = time.perf_counter()
    opt.optimise_stochastic(restarts=20, max_iter=5)
    t_stoch = time.perf_counter() - start
    np.random.seed(1)
    start = time.perf_counter()
    opt.optimise_sa(start_temp=1000.0, cooling=0.99, inner_iter=200)
    t_sa = time.perf_counter() - start
    assert t_stoch < t_sa
