"""Bayesian belief updating for a jar-and-ball experiment with draws without replacement."""

from __future__ import annotations

from math import comb
from typing import List, Optional

import numpy as np


def hypergeometric_pmf(n_red_jar: int, n_balls: int, n_draws: int, k_red: int) -> float:
    """
    P(observe k_red red | n_red_jar red in jar, n_balls total, n_draws draws without replacement).
    Uses the hypergeometric distribution.
    """
    k_blue = n_draws - k_red
    n_blue_jar = n_balls - n_red_jar
    if k_red < 0 or k_blue < 0 or k_red > n_red_jar or k_blue > n_blue_jar:
        return 0.0
    return comb(n_red_jar, k_red) * comb(n_blue_jar, k_blue) / comb(n_balls, n_draws)


class BeliefEngine:
    """
    Tracks an agent's beliefs about the number of red balls in a jar of n_balls total.

    Hypotheses h ∈ {0, 1, …, n_balls} represent the possible counts of red balls.

    Each call to update() or update_asymmetric() blends the Bayesian posterior
    with the pre-update belief using a bias parameter λ:

        posterior = (1 − λ) · bayes_posterior + λ · current_belief

    λ=0 → perfectly Bayesian; λ=1 → no updating at all.

    update_asymmetric() accepts separate λ values for confirming vs. disconfirming
    evidence (per-hypothesis), enabling confirmation-bias modelling.
    """

    def __init__(
        self,
        n_balls: int = 20,
        hypotheses: Optional[List[int]] = None,
        prior: Optional[np.ndarray] = None,
    ) -> None:
        self.n_balls = n_balls
        self.hypotheses: List[int] = (
            hypotheses if hypotheses is not None else list(range(n_balls + 1))
        )
        n_h = len(self.hypotheses)
        if prior is None:
            initial = np.ones(n_h) / n_h
        else:
            arr = np.asarray(prior, dtype=float)
            if arr.shape != (n_h,):
                raise ValueError(f"prior must have length {n_h}, got {arr.shape}")
            if arr.sum() == 0:
                raise ValueError("prior must not be all zeros")
            initial = arr / arr.sum()
        self._initial_prior: np.ndarray = initial
        self.belief: np.ndarray = initial.copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _likelihoods(self, n_draws: int, k_red: int) -> np.ndarray:
        return np.array(
            [hypergeometric_pmf(h, self.n_balls, n_draws, k_red) for h in self.hypotheses]
        )

    def _bayes_update(self, current: np.ndarray, n_draws: int, k_red: int) -> np.ndarray:
        unnorm = self._likelihoods(n_draws, k_red) * current
        total = unnorm.sum()
        # Degenerate case: observation impossible under all hypotheses → don't update
        return current.copy() if total == 0.0 else unnorm / total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, n_draws: int, k_red: int, lam: float = 0.0) -> np.ndarray:
        """
        Update beliefs after observing k_red red balls in n_draws draws.

        Parameters
        ----------
        n_draws : number of balls drawn
        k_red   : number of red balls observed
        lam     : bias weight toward current belief in [0, 1]
                  0 → pure Bayes, 1 → no update

        Returns the new belief distribution (also stored in self.belief).
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        current = self.belief
        bayes = self._bayes_update(current, n_draws, k_red)
        biased = (1.0 - lam) * bayes + lam * current
        self.belief = biased / biased.sum()
        return self.belief.copy()

    def update_asymmetric(
        self,
        n_draws: int,
        k_red: int,
        lam_confirm: float = 0.0,
        lam_disconfirm: float = 0.0,
    ) -> np.ndarray:
        """
        Update beliefs with hypothesis-level asymmetric bias.

        For each hypothesis h:
        - If Bayesian posterior(h) ≥ current belief(h)  → apply lam_confirm
        - If Bayesian posterior(h)  < current belief(h)  → apply lam_disconfirm

        Canonical confirmation bias: lam_confirm=0, lam_disconfirm≈1 means the
        agent fully embraces confirming evidence but resists disconfirming evidence.

        Returns the new belief distribution (also stored in self.belief).
        """
        if not 0.0 <= lam_confirm <= 1.0:
            raise ValueError(f"lam_confirm must be in [0, 1], got {lam_confirm}")
        if not 0.0 <= lam_disconfirm <= 1.0:
            raise ValueError(f"lam_disconfirm must be in [0, 1], got {lam_disconfirm}")
        current = self.belief
        bayes = self._bayes_update(current, n_draws, k_red)
        lams = np.where(bayes >= current, lam_confirm, lam_disconfirm)
        biased = (1.0 - lams) * bayes + lams * current
        self.belief = biased / biased.sum()
        return self.belief.copy()

    def quadratic_score(
        self, predicted_probs: np.ndarray, true_hypothesis_idx: int
    ) -> float:
        """
        Quadratic (Brier) scoring rule.

            QS(p, i*) = 2·p[i*] − ‖p‖²

        Range: (−1, 1]. Score equals 1 for a perfectly certain correct prediction.
        This is a proper scoring rule: maximised in expectation by reporting true beliefs.

        Parameters
        ----------
        predicted_probs     : probability distribution over hypotheses
        true_hypothesis_idx : index of the realised hypothesis in self.hypotheses
        """
        p = np.asarray(predicted_probs, dtype=float)
        return float(2.0 * p[true_hypothesis_idx] - np.dot(p, p))

    def reset(self) -> None:
        """Restore belief to the original prior passed at construction."""
        self.belief = self._initial_prior.copy()
