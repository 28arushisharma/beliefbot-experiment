"""
beliefbot
=========
Bayesian and confirmation-biased belief updating for jar-and-ball experiments.

Quick start::

    from beliefbot import BeliefEngine

    engine = BeliefEngine(n_balls=20)
    engine.update_asymmetric(n_draws=3, k_red=2, lam_confirm=0.0, lam_disconfirm=0.6)
    print(engine.belief)
"""

from beliefbot.belief_engine import BeliefEngine, hypergeometric_pmf

__all__ = ["BeliefEngine", "hypergeometric_pmf"]
__version__ = "0.1.0"
