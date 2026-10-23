import numpy as np
import pytest

pytest.importorskip("numpyro")

import jax

from relyflow import make_design
from relyflow.mcmc import _Space, make_log_likelihood, nuts
from relyflow.models import MODELS, get_model


@pytest.mark.parametrize("name", sorted(MODELS))
def test_jax_likelihood_matches_the_numpy_likelihood(name):
    model = get_model(name)
    rng = np.random.default_rng(0)
    stimulus, advice = make_design([(60, None), (80, 0.8), (60, 0.6)], rng)
    truth = model.sample_prior(1, rng)
    choice, rt = model.simulate(truth, stimulus, advice, rng)
    candidates = model.sample_prior(20, rng)
    candidates["sv_match"][0] = 0.1  # push one set deep into the truncation tail
    candidates["inhibition"][0] = 3.5
    expected = model.log_likelihood(candidates, stimulus, advice, choice, rt)
    with jax.enable_x64(True):
        loglik = make_log_likelihood(model, stimulus[0], advice[0], choice[0], rt[0])
        actual = [float(loglik({k: v[i] for k, v in candidates.items()})) for i in range(20)]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_jax_priors_match_scipy(name):
    model = get_model(name)
    draws = model.sample_prior(10, np.random.default_rng(1))
    space = _Space(model)
    expected = sum(p.prior.logpdf(draws[p.name]) for p in model.parameters)
    with jax.enable_x64(True):
        actual = np.array([float(space.log_prior({k: v[i] for k, v in draws.items()}))
                           for i in range(10)])  # fmt: skip
    # JAX's beta log-density differs from SciPy's by ~1e-6 (float precision).
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


def test_gradients_are_finite_including_the_tail():
    model = get_model("learning")
    rng = np.random.default_rng(2)
    stimulus, advice = make_design([(40, None), (60, 0.8)], rng)
    choice, rt = model.simulate(model.sample_prior(1, rng), stimulus, advice, rng)
    params = {k: float(v[0]) for k, v in model.sample_prior(1, rng).items()}
    params.update(sv_match=0.1, inhibition=3.5, v_match_0=0.2, v_match_1=0.2)
    with jax.enable_x64(True):
        grad = jax.grad(make_log_likelihood(model, stimulus[0], advice[0], choice[0], rt[0]))(
            params
        )
    assert all(np.isfinite(float(g)) for g in grad.values())


@pytest.mark.slow
def test_nuts_recovers_a_simulated_operator():
    model = get_model("reliance")
    rng = np.random.default_rng(3)
    truth = model.sample_prior(1, rng)
    stimulus, advice = make_design([(200, None), (400, 0.85)], rng)
    choice, rt = model.simulate(truth, stimulus, advice, rng)
    result = nuts(model, stimulus[0], advice[0], choice[0], rt[0], seed=4)
    assert result.converged and result.divergences == 0
    draws = result.flat()
    inside = [np.quantile(draws[n], 0.005) <= truth[n][0] <= np.quantile(draws[n], 0.995)
              for n in model.param_names]  # fmt: skip
    assert sum(inside) >= len(inside) - 1
