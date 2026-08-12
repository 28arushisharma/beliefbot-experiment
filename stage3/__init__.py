from otree.api import *
import numpy as np
import json

from beliefbot import BeliefEngine, hypergeometric_pmf  # noqa: F401

doc = """
Stage 3 — Conformity (Coordination Game).

12 rounds: 4 matches × 3 sub-rounds each.
One jar for all 12 rounds (Red or Blue, assigned from session code).
10 balls drawn per sub-round with replacement.

Matches 1-2 (rounds 1-6):  vs computer bot (fixed random choice).
Matches 3-4 (rounds 7-12): vs real human partner.

Payoffs per sub-round:
  Both correct:  $30
  Both wrong:    $20
  One correct:   $10 for correct / $0 for wrong
"""

PAYOFF_BOTH_CORRECT = 30
PAYOFF_BOTH_WRONG   = 20
PAYOFF_ONE_CORRECT  = 10
PAYOFF_ONE_WRONG    = 0


class C(BaseConstants):
    NAME_IN_URL       = 'stage3'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS        = 12
    ROUNDS_PER_MATCH  = 3
    NUM_MATCHES       = 4
    BOT_MATCHES       = 2   # matches 1-2 vs bot; matches 3-4 vs human

    N_BALLS      = 20
    N_DRAWS      = 10
    RED_JAR_RED  = 14
    RED_JAR_BLUE = 6
    BLUE_JAR_RED = 6
    BLUE_JAR_BLUE = 14

    PAYOFF_BOTH_CORRECT = PAYOFF_BOTH_CORRECT
    PAYOFF_BOTH_WRONG   = PAYOFF_BOTH_WRONG
    PAYOFF_ONE_CORRECT  = PAYOFF_ONE_CORRECT
    PAYOFF_ONE_WRONG    = PAYOFF_ONE_WRONG


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    draw_red         = models.IntegerField(initial=0)
    draw_blue        = models.IntegerField(initial=0)
    balls_drawn_json = models.LongStringField(blank=True)


class Player(BasePlayer):
    jar_assignment    = models.StringField()
    guess             = models.StringField(
        choices=[['Red',  'Red Jar (14 red, 6 blue)'],
                 ['Blue', 'Blue Jar (14 blue, 6 red)']],
        label='Which jar do you think these balls came from?',
        widget=widgets.RadioSelect,
        blank=True,
    )
    is_correct        = models.BooleanField(initial=False)
    payoff_this_round = models.FloatField(initial=0.0)
    posterior_red     = models.FloatField(initial=0.5)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed_jar(session_code):
    return abs(hash((session_code, 'stage3_jar'))) % (2 ** 31)

def _seed_draw(session_code, round_number):
    return abs(hash((session_code, 'stage3_draw', round_number))) % (2 ** 31)

def _seed_bot(session_code, group_id):
    return abs(hash((session_code, 'stage3_bot', group_id))) % (2 ** 31)


def _get_jar(session_code):
    """Single jar for all 12 rounds, determined by session code."""
    rng = np.random.default_rng(_seed_jar(session_code))
    return 'Red' if rng.integers(0, 2) == 0 else 'Blue'


def _get_draw(session_code, round_number):
    """Return (draw_red, draw_blue) for N_DRAWS with-replacement draws."""
    jar   = _get_jar(session_code)
    rng   = np.random.default_rng(_seed_draw(session_code, round_number))
    n_red = C.RED_JAR_RED if jar == 'Red' else C.BLUE_JAR_RED
    k_red = int(rng.hypergeometric(n_red, C.N_BALLS - n_red, C.N_DRAWS))
    return k_red, C.N_DRAWS - k_red


def _get_bot_choice(session_code, group_id):
    """Fixed bot choice for a given group: Red or Blue."""
    rng = np.random.default_rng(_seed_bot(session_code, group_id))
    return 'Red' if rng.integers(0, 2) == 0 else 'Blue'


def _match_number(round_number):
    return (round_number - 1) // C.ROUNDS_PER_MATCH + 1

def _round_in_match(round_number):
    return (round_number - 1) % C.ROUNDS_PER_MATCH + 1

def _is_bot_match(round_number):
    return _match_number(round_number) <= C.BOT_MATCHES

def _is_bot_player(player):
    """Player 2 in bot matches acts as computer opponent; all their pages are hidden."""
    return _is_bot_match(player.round_number) and player.id_in_group == 2

def _match_start_round(match_number):
    return (match_number - 1) * C.ROUNDS_PER_MATCH + 1

def _match_end_round(match_number):
    return match_number * C.ROUNDS_PER_MATCH


def _posterior(draw_red):
    """P(Red Jar | draw_red out of N_DRAWS balls), per-round."""
    engine = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine.update(n_draws=C.N_DRAWS, k_red=draw_red)
    return float(engine.belief[1])


# ── Session creation ──────────────────────────────────────────────────────────

def creating_session(subsession):
    for player in subsession.get_players():
        player.jar_assignment = _get_jar(player.session.code)
        if subsession.round_number == 1 and player.id_in_group == 2:
            bot_choice = _get_bot_choice(
                subsession.session.code,
                player.group.id_in_subsession,
            )
            player.participant.vars['bot_choice'] = bot_choice


# ── Pages ─────────────────────────────────────────────────────────────────────

class StageIntroPage(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1 and not _is_bot_player(player)

    @staticmethod
    def vars_for_template(player):
        return dict(
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class MatchTransitionPage(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number in (4, 7, 10) and not _is_bot_player(player)

    @staticmethod
    def vars_for_template(player):
        match      = _match_number(player.round_number)
        prev_match = match - 1
        prev_start = _match_start_round(prev_match)
        prev_end   = _match_end_round(prev_match)
        # Player 2 was a bot in matches 1-2; those payoffs don't count for them
        was_bot_prev = _is_bot_match(prev_start) and player.id_in_group == 2
        if was_bot_prev:
            prev_match_net = 0
        else:
            prev_match_net = int(sum(
                player.in_round(r).payoff_this_round
                for r in range(prev_start, prev_end + 1)
            ))
        return dict(
            match_number=match,
            prev_match_number=prev_match,
            prev_match_net=prev_match_net,
            is_vs_human=not _is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class DrawPage(Page):
    @staticmethod
    def is_displayed(player):
        return not _is_bot_player(player)

    @staticmethod
    def vars_for_template(player):
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)
        return dict(
            draw_red=draw_red,
            draw_blue=draw_blue,
            red_balls=list(range(draw_red)),
            blue_balls=list(range(draw_blue)),
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class ChoicePage(Page):
    form_model  = 'player'
    form_fields = ['guess']

    @staticmethod
    def error_message(player, values):
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def is_displayed(player):
        return not _is_bot_player(player)

    @staticmethod
    def vars_for_template(player):
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)
        return dict(
            draw_red=draw_red,
            draw_blue=draw_blue,
            red_balls=list(range(draw_red)),
            blue_balls=list(range(draw_blue)),
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        # In bot rounds, ResultsWaitPage is skipped entirely (is_displayed=False).
        # P1 therefore computes and stores all group results here immediately after
        # submitting their guess — no cross-player synchronisation is needed.
        if not (_is_bot_match(player.round_number) and player.id_in_group == 1):
            return

        p2  = player.group.get_player_by_id(2)
        jar = _get_jar(player.session.code)
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)

        player.group.draw_red         = draw_red
        player.group.draw_blue        = draw_blue
        player.group.balls_drawn_json = json.dumps(['R'] * draw_red + ['B'] * draw_blue)

        p2.guess          = p2.participant.vars.get('bot_choice', 'Red')
        player.is_correct = (player.guess == jar)
        p2.is_correct     = (p2.guess == jar)

        both_correct = player.is_correct and p2.is_correct
        both_wrong   = not player.is_correct and not p2.is_correct

        if both_correct:
            player.payoff_this_round = float(PAYOFF_BOTH_CORRECT)
            p2.payoff_this_round     = float(PAYOFF_BOTH_CORRECT)
        elif both_wrong:
            player.payoff_this_round = float(PAYOFF_BOTH_WRONG)
            p2.payoff_this_round     = float(PAYOFF_BOTH_WRONG)
        else:
            player.payoff_this_round = float(PAYOFF_ONE_CORRECT if player.is_correct else PAYOFF_ONE_WRONG)
            p2.payoff_this_round     = float(PAYOFF_ONE_CORRECT if p2.is_correct else PAYOFF_ONE_WRONG)

        post = _posterior(draw_red)
        player.posterior_red = post
        p2.posterior_red     = post

        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.payoff_this_round


class ResultsWaitPage(WaitPage):
    @staticmethod
    def is_displayed(player):
        # Bot rounds (1-6): WaitPage is skipped for all players.
        # Results are computed in ChoicePage.before_next_page for P1 instead,
        # because P2 (bot) has no browser and can never arrive here.
        # Human rounds (7-12): both players must arrive before results are shown.
        return not _is_bot_match(player.round_number)

    @staticmethod
    def after_all_players_arrive(group):
        # Only called in human rounds (7-12).
        p1  = group.get_player_by_id(1)
        p2  = group.get_player_by_id(2)
        jar = _get_jar(group.session.code)
        draw_red, draw_blue = _get_draw(group.session.code, group.round_number)

        group.draw_red         = draw_red
        group.draw_blue        = draw_blue
        group.balls_drawn_json = json.dumps(['R'] * draw_red + ['B'] * draw_blue)

        if not p1.guess or not p2.guess:
            raise ValueError(
                f"Stage 3 round {group.round_number}: missing guess in human match "
                f"(p1.guess={p1.guess!r}, p2.guess={p2.guess!r})"
            )

        p1.is_correct = (p1.guess == jar)
        p2.is_correct = (p2.guess == jar)

        both_correct = p1.is_correct and p2.is_correct
        both_wrong   = not p1.is_correct and not p2.is_correct

        if both_correct:
            p1.payoff_this_round = float(PAYOFF_BOTH_CORRECT)
            p2.payoff_this_round = float(PAYOFF_BOTH_CORRECT)
        elif both_wrong:
            p1.payoff_this_round = float(PAYOFF_BOTH_WRONG)
            p2.payoff_this_round = float(PAYOFF_BOTH_WRONG)
        else:
            p1.payoff_this_round = float(PAYOFF_ONE_CORRECT if p1.is_correct else PAYOFF_ONE_WRONG)
            p2.payoff_this_round = float(PAYOFF_ONE_CORRECT if p2.is_correct else PAYOFF_ONE_WRONG)

        post = _posterior(draw_red)
        p1.posterior_red = post
        p2.posterior_red = post

        for p in [p1, p2]:
            prev = p.participant.vars.get('cumulative_earnings', 0)
            p.participant.vars['cumulative_earnings'] = prev + p.payoff_this_round


class ResultsPage(Page):
    @staticmethod
    def is_displayed(player):
        return not _is_bot_player(player)

    @staticmethod
    def vars_for_template(player):
        partner = player.group.get_player_by_id(3 - player.id_in_group)
        match   = _match_number(player.round_number)
        rig     = _round_in_match(player.round_number)
        return dict(
            draw_red=player.group.draw_red,
            draw_blue=player.group.draw_blue,
            red_balls=list(range(player.group.draw_red)),
            blue_balls=list(range(player.group.draw_blue)),
            jar_assignment=player.jar_assignment,
            match_number=match,
            round_in_match=rig,
            is_bot_match=_is_bot_match(player.round_number),
            guess=player.guess,
            is_correct=player.is_correct,
            payoff_this_round=int(player.payoff_this_round),
            partner_guess=partner.guess,
            partner_is_correct=partner.is_correct,
            is_last_round=player.round_number == C.NUM_ROUNDS,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


page_sequence = [
    StageIntroPage,
    MatchTransitionPage,
    DrawPage,
    ChoicePage,
    ResultsWaitPage,
    ResultsPage,
]
