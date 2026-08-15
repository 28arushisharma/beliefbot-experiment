from otree.api import *
import numpy as np
import json

from beliefbot import BeliefEngine, hypergeometric_pmf

doc = """
Stage 1 — Bayesian Learning.

6 rounds split into two jar groups of 3 rounds each.
Rounds 1-3: jar group 1 (Red or Blue, randomly assigned).
Rounds 4-6: jar group 2 (always the opposite colour of group 1).

Within each group, draws are WITH replacement — every round draws 6 fresh balls
from the full 20-ball jar.  The Bayesian posterior resets to a uniform prior
at the start of each new jar group.

Payoff: $5 correct, $0 wrong per round.
"""

PAYOFF_CORRECT = 5  # dollars (placeholder)


class C(BaseConstants):
    NAME_IN_URL       = 'stage1'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS        = 6

    ROUNDS_PER_JAR = 3   # rounds 1-3 = jar group 1, rounds 4-6 = jar group 2

    N_BALLS = 20
    N_DRAWS = 6

    RED_JAR_RED   = 14   # Red Jar:  14 red,  6 blue
    RED_JAR_BLUE  = 6
    BLUE_JAR_RED  = 6    # Blue Jar:  6 red, 14 blue
    BLUE_JAR_BLUE = 14

    PAYOFF_CORRECT = PAYOFF_CORRECT  # mirrored for template access via C.PAYOFF_CORRECT


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    jar_group      = models.IntegerField()     # 1 (rounds 1-3) or 2 (rounds 4-6)
    jar_assignment = models.StringField()      # 'Red' or 'Blue'
    draw_red       = models.IntegerField()
    draw_blue      = models.IntegerField()
    balls_drawn    = models.LongStringField()  # JSON list of 'R'/'B', one per ball
    guess          = models.StringField(
        choices=[
            ['Red',  'Red Jar (14 red, 6 blue)'],
            ['Blue', 'Blue Jar (14 blue, 6 red)'],
        ],
        label='Which jar do you think these balls came from?',
        widget=widgets.RadioSelect,
    )
    is_correct        = models.BooleanField()
    payoff_this_round = models.FloatField()
    posterior_red     = models.FloatField()    # P(Red Jar | draws within current jar group)


# ── Helpers ────────────────────────────────────────────────────────────────

def _seed(session_code: str) -> int:
    return abs(hash(session_code)) % (2 ** 31)


def _jar_group(round_number: int) -> int:
    """Return 1 for rounds 1-3, 2 for rounds 4-6."""
    return 1 if round_number <= C.ROUNDS_PER_JAR else 2


def _get_jar(session_code: str, jar_group: int) -> str:
    """
    Jar group 1: randomly Red or Blue (seeded from session code).
    Jar group 2: always the opposite of group 1 so participants see both jar types.
    """
    rng = np.random.default_rng(_seed(session_code))
    jar_1 = 'Red' if rng.integers(0, 2) == 0 else 'Blue'
    if jar_group == 1:
        return jar_1
    return 'Blue' if jar_1 == 'Red' else 'Red'


def _get_draw(session_code: str, round_number: int) -> tuple[int, int]:
    """
    Return (draw_red, draw_blue) for the given round.

    Draws are WITH REPLACEMENT: every round samples 6 balls from the full
    20-ball jar independently.  The RNG advances sequentially across all 6
    rounds so each round gets a different draw.  Calling this for the same
    (session_code, round_number) always returns the same answer.
    """
    rng = np.random.default_rng(_seed(session_code))
    rng.integers(0, 2)  # consume jar-group-1 assignment draw; keeps round draws in sync

    for r in range(1, round_number + 1):
        jar   = _get_jar(session_code, _jar_group(r))
        n_red = C.RED_JAR_RED  if jar == 'Red' else C.BLUE_JAR_RED
        n_blue = C.N_BALLS - n_red
        k_red = int(rng.hypergeometric(n_red, n_blue, C.N_DRAWS))
        if r == round_number:
            return k_red, C.N_DRAWS - k_red


def _posterior_red(session_code: str, round_number: int) -> float:
    """
    Cumulative Bayesian P(Red Jar | draws within the current jar group).

    Prior resets to uniform at the start of each new jar group (round 1 and round 4).
    """
    group      = _jar_group(round_number)
    first_round = 1 if group == 1 else C.ROUNDS_PER_JAR + 1
    engine = BeliefEngine(
        n_balls=C.N_BALLS,
        hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED],  # belief[1] = P(Red Jar)
    )
    for r in range(first_round, round_number + 1):
        k_red, _ = _get_draw(session_code, r)
        engine.update(n_draws=C.N_DRAWS, k_red=k_red)
    return float(engine.belief[1])


# ── Instructions widget ────────────────────────────────────────────────────

def _instructions_vars():
    return dict(
        stage_instructions_bullets=[
            'Observe 6 balls drawn from a 20-ball jar (with replacement).',
            'Guess: Red Jar (14 red, 6 blue) or Blue Jar (14 blue, 6 red)?',
            'Rounds 1–3 use one jar; rounds 4–6 switch to the opposite jar.',
            'Earn $5 per correct guess, $0 for wrong.',
        ],
        show_payoff_table=False,
    )


# ── Session creation ───────────────────────────────────────────────────────

def creating_session(subsession: Subsession):
    """Set jar_group and jar_assignment for every player in every round."""
    for player in subsession.get_players():
        g = _jar_group(player.round_number)
        player.jar_group      = g
        player.jar_assignment = _get_jar(player.session.code, g)


# ── Pages ─────────────────────────────────────────────────────────────────

class StageIntroPage(Page):
    """Transition screen shown once at the very start (round 1 only)."""

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        return dict(
            **_instructions_vars(),
            rounds_per_jar_plus_1=C.ROUNDS_PER_JAR + 1,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class JarChangePage(Page):
    """Transition shown at round 4: the jar has changed."""

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.ROUNDS_PER_JAR + 1

    @staticmethod
    def vars_for_template(player: Player):
        group1_total = int(sum(
            player.in_round(r).payoff_this_round
            for r in range(1, C.ROUNDS_PER_JAR + 1)
        ))
        return dict(
            **_instructions_vars(),
            group1_total          = group1_total,
            group1_max            = PAYOFF_CORRECT * C.ROUNDS_PER_JAR,
            rounds_per_jar_plus_1 = C.ROUNDS_PER_JAR + 1,
            cumulative_earnings   = int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class IntroPage(Page):
    """Display this round's 6-ball draw as colored circles. Store draw data on submit."""

    @staticmethod
    def vars_for_template(player: Player):
        k_red, k_blue = _get_draw(player.session.code, player.round_number)
        group          = player.jar_group
        group_start    = 1 if group == 1 else C.ROUNDS_PER_JAR + 1
        group_end      = C.ROUNDS_PER_JAR if group == 1 else C.NUM_ROUNDS
        return dict(
            **_instructions_vars(),
            draw_red    = k_red,
            draw_blue   = k_blue,
            red_balls   = list(range(k_red)),
            blue_balls  = list(range(k_blue)),
            group_start = group_start,
            group_end   = group_end,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        k_red, k_blue = _get_draw(player.session.code, player.round_number)
        player.draw_red    = k_red
        player.draw_blue   = k_blue
        player.balls_drawn = json.dumps(['R'] * k_red + ['B'] * k_blue)


class ChoicePage(Page):
    """Participant guesses which jar the balls came from."""

    form_model  = 'player'
    form_fields = ['guess']

    @staticmethod
    def error_message(player, values):
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def vars_for_template(player: Player):
        k_red, k_blue = _get_draw(player.session.code, player.round_number)
        return dict(
            **_instructions_vars(),
            draw_red   = k_red,
            draw_blue  = k_blue,
            red_balls  = list(range(k_red)),
            blue_balls = list(range(k_blue)),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        player.is_correct        = (player.guess == player.jar_assignment)
        player.payoff_this_round = float(PAYOFF_CORRECT) if player.is_correct else 0.0
        player.posterior_red     = _posterior_red(player.session.code, player.round_number)
        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.payoff_this_round


class ResultsPage(Page):
    """Show correctness, round payoff, and running totals."""

    @staticmethod
    def vars_for_template(player: Player):
        k_red, k_blue = _get_draw(player.session.code, player.round_number)
        group         = player.jar_group
        round_in_group = (player.round_number if group == 1
                          else player.round_number - C.ROUNDS_PER_JAR)

        overall_total = int(sum(
            player.in_round(r).payoff_this_round
            for r in range(1, player.round_number + 1)
        ))
        group_start = 1 if group == 1 else C.ROUNDS_PER_JAR + 1
        group_total = int(sum(
            player.in_round(r).payoff_this_round
            for r in range(group_start, player.round_number + 1)
        ))
        return dict(
            **_instructions_vars(),
            draw_red           = k_red,
            draw_blue          = k_blue,
            red_balls          = list(range(k_red)),
            blue_balls         = list(range(k_blue)),
            jar_assignment     = player.jar_assignment,
            jar_group          = group,
            round_in_group     = round_in_group,
            guess              = player.guess,
            is_correct         = player.is_correct,
            payoff_this_round  = int(player.payoff_this_round),
            overall_total      = overall_total,
            overall_max        = PAYOFF_CORRECT * player.round_number,
            group_total        = group_total,
            group_max          = PAYOFF_CORRECT * round_in_group,
            is_last_round      = player.round_number == C.NUM_ROUNDS,
            cumulative_earnings= int(player.participant.vars.get('cumulative_earnings', 0)),
        )


page_sequence = [StageIntroPage, JarChangePage, IntroPage, ChoicePage, ResultsPage]
