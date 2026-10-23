"""Benchmark amortized posteriors against NUTS for accuracy and speed."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from relyflow import TrialData, _workers
from relyflow.inference.amortizer import Amortizer
from relyflow.inference.refine import importance_resample

REFERENCE_SETTINGS = {"num_chains": 4, "num_warmup": 1000, "num_samples": 1000}


def compare(draws: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-parameter agreement with a reference posterior, in units of the reference SD.

    - ``mean_error``: (mean - reference mean) / reference SD
    - ``sd_ratio``: SD / reference SD (above 1: too wide; below 1: overconfident)
    - ``wasserstein``: 1-D Wasserstein distance / reference SD
    """
    rows = []
    for name, ref in reference.items():
        x, sd = draws[name], ref.std()
        rows.append(
            {
                "parameter": name,
                "mean_error": (x.mean() - ref.mean()) / sd,
                "sd_ratio": x.std() / sd,
                "wasserstein": wasserstein_distance(x, ref) / sd,
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(
    amortizer: Amortizer,
    operators: list[TrialData],
    seed: int = 0,
    workers: int = 10,
    reference_settings: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare network and importance-resampled posteriors with NUTS references.

    NUTS runs from prior draws in single-threaded worker processes, one operator per
    worker, so its timings are single-core seconds.
    Returns ``(agreement, timing)``: agreement per operator, method and parameter;
    timing per operator and method in single-core seconds, plus NUTS diagnostics.
    """
    settings = reference_settings or REFERENCE_SETTINGS
    model = amortizer.model

    # Amortized methods first, timed without competition from the NUTS workers.
    methods: list[dict] = []
    for i, data in enumerate(operators):
        start = time.perf_counter()
        network = amortizer.sample(data, num_samples=4000, seed=seed + i)
        network_seconds = time.perf_counter() - start
        start = time.perf_counter()
        refined = importance_resample(amortizer, data, num_samples=4000, seed=seed + i)
        methods.append(
            {
                "network": (network, network_seconds),
                "importance": (refined.draws, time.perf_counter() - start),
                "efficiency": refined.diagnostics["efficiency"],
            }
        )

    context = multiprocessing.get_context("spawn")
    with (
        _workers.single_threaded_environment(),
        ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool,
    ):
        futures = [
            pool.submit(
                _workers.nuts_reference,
                model.name,
                (data.stimulus, data.advice, data.choice, data.rt),
                seed + i,
                settings,
            )
            for i, data in enumerate(operators)
        ]
        references = [future.result() for future in futures]

    agreement, timing = [], []
    for i, (reference, result) in enumerate(zip(references, methods, strict=True)):
        for method in ("network", "importance"):
            draws, seconds = result[method]
            table = compare(draws, reference["draws"])
            table.insert(0, "method", method)
            table.insert(0, "operator", i)
            agreement.append(table)
            timing.append({"operator": i, "method": method, "seconds": seconds})
        timing.append({"operator": i, "method": "nuts", "seconds": reference["seconds"]})
        timing[-1].update(
            max_rhat=reference["max_rhat"],
            min_ess=reference["min_ess"],
            divergences=reference["divergences"],
            efficiency=result["efficiency"],
        )
    return pd.concat(agreement, ignore_index=True), pd.DataFrame(timing)
