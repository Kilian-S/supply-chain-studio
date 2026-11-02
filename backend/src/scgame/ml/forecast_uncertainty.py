"""Uncertainty and coherence for the demand forecast: sample paths from past errors, reconciliation,
quantiles and the scores.

Past errors are one-year-ahead errors at rolling origins inside the fitting window, as log((y + 1) /
(forecast + 1)) per series and month. Each block (one origin, all series, twelve months) splits into a
level error (the block's mean per series) and month noise (the rest). A sample path draws one block per
forecast year for all series together; level errors add up over the years, month noise does not. A
width factor k, chosen on the validation set, scales both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)
OFFSET = 1.0  # units added before taking logs, so empty months have an error


@dataclass
class ErrorBlocks:
    level: np.ndarray  # (origin, series)
    noise: np.ndarray  # (origin, series, 12)


def error_blocks(actual: np.ndarray, forecast: np.ndarray) -> ErrorBlocks:
    """`actual` and `forecast` are (series, origin, 12)."""
    r = np.log((actual + OFFSET) / (forecast + OFFSET))
    lvl = r.mean(axis=2)
    return ErrorBlocks(level=lvl.T.copy(), noise=np.transpose(r - lvl[:, :, None], (1, 0, 2)).copy())


def draw_origins(n_origins: int, years: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_origins, size=(n_paths, years))


def sample_paths(mean: np.ndarray, blocks: ErrorBlocks, draws: np.ndarray, k: float) -> np.ndarray:
    """Paths (path, series, months) around mean forecasts (series, months), with the drawn origins."""
    n_paths, years = draws.shape
    n, H = mean.shape
    out = np.empty((n_paths, n, H), dtype=np.float32)
    cum = np.zeros((n_paths, n), dtype=np.float32)
    for j in range(min(years, H // 12)):
        o = draws[:, j]
        cum += blocks.level[o]
        logm = k * (cum[:, :, None] + blocks.noise[o])
        out[:, :, 12 * j:12 * (j + 1)] = (mean[None, :, 12 * j:12 * (j + 1)] + OFFSET) * np.exp(logm) - OFFSET
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------- reconciliation


def shrunk_covariance(e: np.ndarray) -> np.ndarray:
    """Covariance of errors (observation, series) shrunk towards its diagonal (Schäfer and Strimmer 2005,
    as used for MinT by Wickramasuriya, Athanasopoulos and Hyndman 2019)."""
    n = e.shape[0]
    e = e - e.mean(0)
    sd = e.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    x = e / sd
    R = x.T @ x / n
    w = np.einsum("ti,tj->tij", x, x)
    var_r = n / (n - 1) ** 2 * ((w - R[None]) ** 2).sum(0)  # variance of the sample correlations
    off = ~np.eye(len(R), dtype=bool)
    lam = float(np.clip(var_r[off].sum() / max((R[off] ** 2).sum(), 1e-12), 0, 1))
    Rs = (1 - lam) * R
    np.fill_diagonal(Rs, 1.0)
    cov = Rs * np.outer(sd, sd)
    return cov + np.eye(len(cov)) * 1e-9 * np.trace(cov) / len(cov)


def mint_matrix(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """P = S (S' W^-1 S)^-1 S' W^-1: maps base forecasts of every series to coherent ones."""
    Wi = np.linalg.pinv(W)
    G = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
    return S @ G


def bottom_up_matrix(S: np.ndarray, bottom_rows: np.ndarray) -> np.ndarray:
    P = np.zeros((S.shape[0], S.shape[0]))
    P[:, bottom_rows] = S
    return P


def reconcile(P: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply P to forecasts (series, months) or paths (path, series, months)."""
    if x.ndim == 2:
        return P @ x
    return np.einsum("ij,pjm->pim", P, x, optimize=True).astype(np.float32)


def middle_out(pairs: np.ndarray, parent: np.ndarray, rc_target: np.ndarray, n_rc: int) -> np.ndarray:
    """Scale Region × SKU forecasts so they sum to their (reconciled) Region × category forecast. Works on
    (series, months) or (path, series, months); the series axis is the second from last. A parent whose
    children forecast nothing keeps them at zero."""
    sums = sum_to(pairs, parent, n_rc)
    ratio = np.where(sums > 0, rc_target / np.where(sums > 0, sums, 1), 1.0)
    return pairs * np.take(ratio, parent, axis=-2)


def sum_to(child: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    """Sum the series axis (second from last) of `child` into `n` groups by `index`."""
    out = np.zeros(child.shape[:-2] + (n, child.shape[-1]), dtype=child.dtype)
    for i in range(n):
        out[..., i, :] = child[..., index == i, :].sum(axis=-2)
    return out


# ---------------------------------------------------------------- scores


def quantiles(paths: np.ndarray) -> np.ndarray:
    """(3, series, months) P10, P50, P90 from (path, series, months)."""
    return np.quantile(paths, QUANTILES, axis=0)


def scores(actual: np.ndarray, q: np.ndarray) -> dict:
    """Scores over a block of series and months. Weighted quantile loss = mean over P10, P50, P90 of
    2 x summed pinball loss / summed actual units (at P50 it equals WAPE)."""
    tot = actual.sum()
    if tot <= 0:
        return {"wql": np.nan, "wape": np.nan, "bias": np.nan, "coverage": np.nan}
    wql = []
    for qi, tau in enumerate(QUANTILES):
        d = actual - q[qi]
        wql.append(2 * np.sum(np.maximum(tau * d, (tau - 1) * d)) / tot)
    return {
        "wql": float(np.mean(wql)),
        "wape": float(np.abs(actual - q[1]).sum() / tot),
        "bias": float((q[1] - actual).sum() / tot),
        "coverage": float(np.mean((actual >= q[0]) & (actual <= q[2]))),
    }


def wape(actual: np.ndarray, point: np.ndarray) -> float:
    tot = actual.sum()
    return float(np.abs(actual - point).sum() / tot) if tot > 0 else np.nan
