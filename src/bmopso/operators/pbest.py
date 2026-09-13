"""Personal best (pbest) update operators following Coello Coello et al. (2004)."""

from __future__ import annotations

from typing import Tuple
import numpy as np

__all__ = ["update_personal_bests"]


def update_personal_bests(
    pbest_x: np.ndarray,
    pbest_f: np.ndarray,
    pbest_cv: np.ndarray | None,
    x: np.ndarray,
    f: np.ndarray,
    cv: np.ndarray,
    random_state: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update personal best positions using Constrained-Dominance and Coello Coello (2004) rules.

    Following Coello Coello et al. (2004):
    1. If current position strictly dominates pbest in memory: pbest = current position.
    2. If pbest strictly dominates current position: keep pbest.
    3. If neither of them is dominated by the other (mutually non-dominated / incomparable):
       one of them is chosen randomly with equal probability (p = 0.5).

    Parameters
    ----------
    pbest_x : np.ndarray
        Current personal best binary positions (n_particles, n_var).
    pbest_f : np.ndarray
        Current personal best objectives (n_particles, n_obj).
    pbest_cv : np.ndarray | None
        Current personal best constraint violations (n_particles,).
    x : np.ndarray
        New binary positions (n_particles, n_var).
    f : np.ndarray
        New objective values (n_particles, n_obj).
    cv : np.ndarray
        New constraint violations (n_particles,).
    random_state : np.random.Generator | None, default=None
        NumPy Generator used for the incomparable coin-flip. If None, uses an isolated Generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated (pbest_x, pbest_f, pbest_cv).
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    n_particles = len(x)
    new_pbest_x = pbest_x.copy()
    new_pbest_f = pbest_f.copy()
    new_pbest_cv = (
        pbest_cv.copy() if pbest_cv is not None else np.zeros(n_particles, dtype=float)
    )

    cv_new = np.asarray(cv, dtype=float).reshape(-1)
    cv_old = np.asarray(new_pbest_cv, dtype=float).reshape(-1)

    new_viol = cv_new > 0.0
    old_viol = cv_old > 0.0
    both_infeasible = new_viol & old_viol
    both_feasible = ~new_viol & ~old_viol

    new_pareto_old = np.all(f <= new_pbest_f, axis=1) & np.any(f < new_pbest_f, axis=1)
    old_pareto_new = np.all(new_pbest_f <= f, axis=1) & np.any(new_pbest_f < f, axis=1)

    new_dom_old = (
        (~new_viol & old_viol)
        | (both_infeasible & (cv_new < cv_old))
        | (both_feasible & new_pareto_old)
    )
    old_dom_new = (
        (~old_viol & new_viol)
        | (both_infeasible & (cv_old < cv_new))
        | (both_feasible & old_pareto_new)
    )

    incomparable = ~new_dom_old & ~old_dom_new
    replace = new_dom_old | (incomparable & (rng.random(n_particles) < 0.5))

    new_pbest_x[replace] = x[replace]
    new_pbest_f[replace] = f[replace]
    new_pbest_cv[replace] = cv_new[replace]

    return new_pbest_x, new_pbest_f, new_pbest_cv
