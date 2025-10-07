#!/usr/bin/env python3
"""
SBPointOptimiser
================
A Synergy-enabled route-optimisation utility for SlideBook.
Travelling-Salesman heuristics to minimise stage motion through all XYZ (and
Aux-Z) points. Supports a classic Nearest-Neighbour + 2-Opt strategy as well
as a slower but sometimes superior simulated-annealing search.

--------------------------------
Pass ``--open_path`` to optimise an open Hamiltonian path (start fixed, end
free).  Omit the flag to keep the classic closed round-trip.

Author : Scott Brooks  
Company: Intelligent Imaging Innovations, Inc.  
Year   : 2025  
License: Proprietary
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from typing import List

import numpy as np

from .microscope_client import MicroscopeClient, Point

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def distance(p1: Point, p2: Point) -> float:
    """
    Euclidean distance between two points (ignoring Aux-Z).

    Args:
        p1, p2: Two microscope points.
    Returns:
        Straight-line distance in microns.
    """
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )

class RouteOptimizer:
    """
    Generate a near-optimal traversal order for a set of points.

    Uses:
      1. Nearest-Neighbour seed (Θ(n²))
      2. Repeated 2-Opt edge swaps (≈ Θ(k·n²))

    Attributes:
        points: Original list of points.
        n:      Number of points.
    """

    def __init__(self, points: List[Point]):
        if len(points) < 2:
            raise ValueError("Need at least two points to optimise a route")
        self.points = points
        self.n = len(points)

        # Pre-compute symmetric distance matrix for O(1) look-ups.
        self._dmat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = distance(points[i], points[j])
                self._dmat[i, j] = self._dmat[j, i] = d

    def optimise(self, max_iter: int = 10) -> List[int]:
        """
        Compute an index order for the shortest *closed* tour.

        Args:
            max_iter: Maximum 2-Opt sweeps.
        Returns:
            List of point indices forming the tour.
        """
        tour = self._nearest_neighbour()
        for _ in range(max_iter):
            improved, tour = self._two_opt(tour)
            if not improved:
                break
        return tour

    def optimise_sa(
        self,
        start_temp: float = 1000.0,
        cooling: float = 0.995,
        inner_iter: int = 100,
    ) -> List[int]:
        """Simulated annealing search for a shorter closed tour.

        Args:
            start_temp: Initial temperature for the annealing schedule.
            cooling:    Multiplicative cooling factor applied after each outer loop.
            inner_iter: Number of candidate swaps per temperature step.

        Returns:
            List of point indices forming the tour.
        """
        tour = self._nearest_neighbour()
        best = tour[:]
        best_len = self._tour_length(best)
        temp = start_temp
        while temp > 1e-3:
            for _ in range(inner_iter):
                i, j = sorted(np.random.choice(range(self.n), 2, replace=False))
                if j - i <= 1:
                    continue  # skip adjacent or identical indices
                candidate = tour[:]
                candidate[i:j] = reversed(candidate[i:j])
                cand_len = self._tour_length(candidate)
                delta = cand_len - self._tour_length(tour)
                if delta < 0 or math.exp(-delta / temp) > np.random.rand():
                    tour = candidate
                    if cand_len < best_len:
                        best, best_len = candidate, cand_len
            temp *= cooling
        return best

    def optimise_stochastic(
        self, restarts: int = 10, max_iter: int = 5
    ) -> List[int]:
        """Iterated stochastic 2-Opt search.

        Repeatedly perturbs the current best tour with a random segment reversal
        followed by local 2-Opt refinement.  Usually finds a better route than
        :meth:`optimise` while remaining much faster than full simulated
        annealing.

        Args:
            restarts: Number of perturbation + refinement cycles to attempt.
            max_iter: Maximum 2-Opt sweeps per refinement step.

        Returns:
            List of point indices forming the tour.
        """
        best = self.optimise(max_iter=max_iter)
        best_len = self._tour_length(best)
        for _ in range(restarts):
            candidate = best[:]
            i, j = sorted(np.random.choice(range(self.n), 2, replace=False))
            if j - i <= 1:
                continue
            candidate[i:j] = reversed(candidate[i:j])
            for _ in range(max_iter):
                improved, candidate = self._two_opt(candidate)
                if not improved:
                    break
            cand_len = self._tour_length(candidate)
            if cand_len < best_len:
                best, best_len = candidate, cand_len
        return best

    def _nearest_neighbour(self) -> List[int]:
        remaining = set(range(self.n))
        current = 0
        tour = [current]
        remaining.remove(current)
        while remaining:
            nxt = min(remaining, key=lambda j: self._dmat[current, j])
            tour.append(nxt)
            remaining.remove(nxt)
            current = nxt
        return tour

    def _two_opt(self, tour: List[int]):
        best_len = self._tour_length(tour)
        n = len(tour)
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # skip adjacent edges
                new_tour = tour[:]
                new_tour[i:j] = reversed(tour[i:j])
                new_len = self._tour_length(new_tour)
                if new_len < best_len - 1e-6:
                    return True, new_tour
        return False, tour

    def _tour_length(self, tour: List[int]) -> float:
        return sum(self._dmat[tour[i], tour[(i + 1) % self.n]] for i in range(self.n))


def main() -> None:
    """Parse CLI, optimise route, and (optionally) upload to SlideBook."""
    parser = argparse.ArgumentParser(
        description="Optimise SlideBook XY-point traversal order"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="SlideBook host (default: 127.0.0.1)")
    parser.add_argument("--port", default=65432, type=int,
                        help="TCP port (default: 65432)")
    parser.add_argument("--max_iter", default=20, type=int,
                        help="Maximum 2-Opt refinement sweeps")
    parser.add_argument("--open_path", action="store_true",
                        help="Allow tour to finish at a different point "
                             "(open Hamiltonian path)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print proposed order instead of uploading")
    args = parser.parse_args()

    with MicroscopeClient(args.host, args.port) as mc:
        points = mc.fetch_points()
        if len(points) < 2:
            logging.info("<2 points – nothing to optimise")
            return

        optimiser = RouteOptimizer(points)
        length_before = optimiser._tour_length(list(range(len(points))))
        start = time.perf_counter()
        closed_order = optimiser.optimise(max_iter=args.max_iter)
        elapsed = time.perf_counter() - start

        # Handle open-path option
        if args.open_path:
            # Remove the single longest edge from the closed tour.
            longest_i = max(
                range(len(closed_order)),
                key=lambda i: distance(points[closed_order[i]],
                                        points[closed_order[(i + 1) %
                                                            len(closed_order)]]))
            order = (closed_order[longest_i + 1:] +
                     closed_order[:longest_i + 1])
            length_after = sum(
                distance(points[a], points[b])
                for a, b in zip(order[:-1], order[1:])
            )
        else:
            order = closed_order
            length_after = optimiser._tour_length(order)

        logging.info(
            "Total path length before: %.1f µm -> after: %.1f µm "
            "(%.1f %% saved, %.2f s)",
            length_before,
            length_after,
            100 * (1 - length_after / length_before),
            elapsed,
        )

        ordered_pts = [points[i] for i in order]

        if args.dry_run:
            for idx, (x, y, z, aux) in enumerate(ordered_pts, 1):
                print(f"{idx:3}: X={x:.1f} µm Y={y:.1f} µm "
                      f"Z={z:.1f} µm AuxZ={aux:.1f}")
        else:
            mc.push_points(ordered_pts)
            logging.info("Optimised route sent to microscope")

if __name__ == "__main__":
    main()
