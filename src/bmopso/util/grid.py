"""Adaptive hypercube grid and leader selection utilities for multiobjective PSO.

Implements the Adaptive Grid mechanism proposed by Carlos A. Coello Coello,
Gregorio Toscano Pulido, and Maximino Salazar Lechuga (2004) in:
"Handling Multiple Objectives With Particle Swarm Optimization", IEEE Transactions
on Evolutionary Computation, Vol. 8, No. 3, pp. 256-279.
DOI: https://doi.org/10.1109/TEVC.2004.826067
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

__all__ = ["AdaptiveGrid"]


class AdaptiveGrid:
    """Adaptive hypercube grid partitioning the objective space into discrete cells.

    Following Coello Coello et al. (2004):
    1. The objective space is bounded by the minimum and maximum objective values of the
       solutions currently in the archive, expanded by an adaptive buffer.
    2. Each objective dimension is subdivided into `n_grid` equal divisions.
    3. Each solution is mapped into a discrete hypercube coordinate (k_1, k_2, ..., k_M)
       where k_j in [0, n_grid - 1].
    4. Occupied hypercubes receive fitness inversely proportional to the number of solutions
       they contain: fitness_i = 10.0 / N_i.
    5. Social leaders (gbest) are selected by performing Roulette Wheel Selection over the
       occupied hypercubes and then picking a solution uniformly at random from the selected hypercube.
    6. When the archive exceeds capacity, solutions are pruned from the most crowded hypercubes.

    Parameters
    ----------
    n_grid : int, default=30
        Number of subdivisions (grid partitions) along each objective dimension.
    """

    def __init__(self, n_grid: int = 30) -> None:
        if n_grid < 1:
            raise ValueError(f"n_grid must be at least 1, got {n_grid}.")
        self.n_grid: int = n_grid

    def compute_grid_coordinates(self, f: np.ndarray) -> np.ndarray:
        """Compute discrete grid coordinates for each solution across all objectives.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix of shape (N, n_obj).

        Returns
        -------
        np.ndarray
            Integer matrix of grid coordinates of shape (N, n_obj), where each entry
            is in [0, n_grid - 1].
        """
        n_points, n_obj = f.shape
        if n_points == 0:
            return np.empty((0, n_obj), dtype=int)

        coords = np.zeros((n_points, n_obj), dtype=int)
        f_min = np.min(f, axis=0)
        f_max = np.max(f, axis=0)

        for m in range(n_obj):
            range_m = float(f_max[m] - f_min[m])
            if range_m == 0.0:
                # If all points share the same objective value, assign to center cell
                coords[:, m] = self.n_grid // 2
            else:
                # Boundary buffer to ensure extreme points lie comfortably inside boundary cells
                buffer = range_m / (2.0 * self.n_grid)
                lower_bound = f_min[m] - buffer
                upper_bound = f_max[m] + buffer
                cell_width = (upper_bound - lower_bound) / self.n_grid

                c = np.floor((f[:, m] - lower_bound) / cell_width).astype(int)
                coords[:, m] = np.clip(c, 0, self.n_grid - 1)

        return coords

    def get_hypercube_ids(self, f: np.ndarray) -> np.ndarray:
        """Compute unique hypercube integer IDs for each solution.

        Uses multidimensional coordinate mapping via np.unique to guarantee
        exact hypercube indexing without integer overflow for any number of objectives.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix of shape (N, n_obj).

        Returns
        -------
        np.ndarray
            1D integer array of shape (N,) containing the hypercube ID of each point.
        """
        n_points, n_obj = f.shape
        if n_points == 0:
            return np.empty((0,), dtype=int)

        coords = self.compute_grid_coordinates(f)
        _, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        return inverse_indices

    def _hypercube_membership(
        self, f: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map solutions to hypercube IDs, occupancy, grouped indices, and group starts.

        Returns
        -------
        hypercube_ids : np.ndarray
            Integer IDs of shape (N,) in ``[0, K)``.
        counts : np.ndarray
            Occupancy of each occupied hypercube, shape (K,).
        order : np.ndarray
            Stable argsort of ``hypercube_ids`` so members of cube ``k`` occupy
            ``order[starts[k] : starts[k] + counts[k]]``.
        starts : np.ndarray
            Start offset of each cube inside ``order``, shape (K,).
        """
        hypercube_ids = self.get_hypercube_ids(f)
        counts = np.bincount(hypercube_ids)
        order = np.argsort(hypercube_ids, kind="stable")
        starts = np.zeros(len(counts), dtype=int)
        if len(counts) > 1:
            starts[1:] = np.cumsum(counts[:-1])
        return hypercube_ids, counts, order, starts

    def _crowded_removal_counts(
        self,
        counts: np.ndarray,
        num_to_remove: int,
        random_state: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Compute per-hypercube removals equivalent to sequential crowded pruning.

        Each step of Coello Coello et al. (2004) deletes one solution from a
        currently most-populated hypercube. Water-filling batches every layer of
        tied maxima: if ``k`` cubes share occupancy ``M`` and the next density is
        ``M2``, the next ``k * (M - M2)`` deletions stay inside that tied set.
        """
        rng = random_state if random_state is not None else np.random.default_rng()
        live = counts.astype(int, copy=True)
        remove_counts = np.zeros_like(live)
        remaining = int(num_to_remove)

        while remaining > 0:
            max_pop = int(live.max())
            at_max = live == max_pop
            n_max = int(np.count_nonzero(at_max))
            if n_max == 0 or max_pop <= 0:
                break

            if np.any(~at_max):
                second = int(live[~at_max].max())
            else:
                second = 0

            layer = max_pop - second
            if layer <= 0:
                break

            batch = min(remaining, n_max * layer)
            per_cube, extra = divmod(batch, n_max)
            delta = np.zeros_like(live)
            delta[at_max] = per_cube
            if extra > 0:
                chosen = rng.choice(
                    np.flatnonzero(at_max), size=extra, replace=False
                )
                delta[chosen] += 1
            live -= delta
            remove_counts += delta
            remaining -= batch

        return remove_counts

    def select_leaders(
        self,
        x: np.ndarray,
        f: np.ndarray,
        n_particles: int,
        random_state: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Select social leaders (gbest) for swarm particles using Coello Coello (2004) Grid Roulette.

        Parameters
        ----------
        x : np.ndarray
            Binary solution positions in the archive of shape (N, n_var).
        f : np.ndarray
            Objective matrix in the archive of shape (N, n_obj).
        n_particles : int
            Number of particles requiring a leader.
        random_state : np.random.Generator | None, default=None
            NumPy Generator used for roulette and within-cube sampling.
            If None, uses an isolated Generator.

        Returns
        -------
        np.ndarray
            Selected binary leader positions of shape (n_particles, n_var).
        """
        n_solutions = len(x)
        if n_solutions == 0:
            raise RuntimeError("Cannot select leaders from an empty archive.")
        if n_solutions == 1:
            return np.tile(x[0], (n_particles, 1))

        rng = random_state if random_state is not None else np.random.default_rng()
        _, counts, order, starts = self._hypercube_membership(f)

        # Fitness is inversely proportional to hypercube population: fitness_i = 10.0 / N_i
        fitnesses = 10.0 / counts.astype(float)
        total_fitness = float(np.sum(fitnesses))

        if total_fitness > 0.0:
            probs = fitnesses / total_fitness
        else:
            probs = np.full(len(counts), 1.0 / len(counts))

        selected_cubes = rng.choice(len(counts), size=n_particles, p=probs)
        offsets = (rng.random(n_particles) * counts[selected_cubes]).astype(int)
        selected_leader_indices = order[starts[selected_cubes] + offsets]
        return x[selected_leader_indices]

    def prune_archive(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: np.ndarray,
        max_size: int,
        random_state: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prune archive solutions when exceeding max capacity by targeting the most crowded hypercubes.

        Following Coello Coello et al. (2004):
        Identifies hypercubes with the highest density (most solutions) and eliminates
        solutions from the most crowded hypercubes until the archive reaches max_size.
        Removals are batched by density layer (water-filling) and are equivalent to
        deleting one random member of a current maximum-occupancy cube at a time.

        Parameters
        ----------
        x : np.ndarray
            Binary solution positions (N, n_var).
        f : np.ndarray
            Objective matrix (N, n_obj).
        cv : np.ndarray
            Constraint violation array (N,).
        max_size : int
            Maximum allowable archive capacity.
        random_state : np.random.Generator | None, default=None
            NumPy Generator used for crowded-cube tie-breaks and victim sampling.
            If None, uses an isolated Generator.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Pruned (x, f, cv) arrays of length max_size.
        """
        num_to_remove = len(x) - max_size
        if num_to_remove <= 0:
            return x, f, cv

        rng = random_state if random_state is not None else np.random.default_rng()
        hypercube_ids, counts, _, _ = self._hypercube_membership(f)
        remove_counts = self._crowded_removal_counts(
            counts, num_to_remove, random_state=rng
        )

        rand_keys = rng.random(len(x))
        prio = np.lexsort((rand_keys, hypercube_ids))
        starts = np.zeros(len(counts), dtype=int)
        if len(counts) > 1:
            starts[1:] = np.cumsum(counts[:-1])
        within_rank = np.arange(len(x)) - np.repeat(starts, counts)
        victims = prio[within_rank < remove_counts[hypercube_ids[prio]]]

        mask = np.ones(len(x), dtype=bool)
        mask[victims] = False
        return x[mask], f[mask], cv[mask]
