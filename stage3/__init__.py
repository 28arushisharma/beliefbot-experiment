from otree.api import *
import numpy as np
import json
from types import SimpleNamespace

from beliefbot import BeliefEngine, hypergeometric_pmf  # noqa: F401

doc = """
Stage 3 — Conformity (Coordination Game).

6 rounds: 2 matches × 3 sub-rounds each.
The jar changes between every match: each match gets a new, independently randomly
assigned jar (Red or Blue), fixed for that match's 3 sub-rounds.
10 balls drawn per sub-round with replacement.

Match 1 (rounds 1-3): every participant plays their own computer (always the bad bot).
Match 2 (rounds 4-6): vs real human partner.
The bot/human distinction is internal only — never shown to participants.

Participants are paired in order at the start of the stage. With an odd number of
participants, the leftover one has no human partner: they play the computer in every
round, instantly (no waiting). When the computer stands in for a missing human
(match 2), it is randomly good or bad.

Good computer: Bayesian (lam_disconfirm=0)      → guesses the higher-posterior jar.
Bad  computer: always guesses the wrong jar (opposite of the true jar) for the whole match.

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
    PLAYERS_PER_GROUP = None   # pairs formed in creating_session; allows an odd leftover
    NUM_ROUNDS        = 6
    ROUNDS_PER_MATCH  = 3
    NUM_MATCHES       = 2
    BOT_MATCHES       = 1   # match 1 vs bot; match 2 vs human

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
    # Seconds between page load and submit (set by JS on every page).
    # time_on_page holds the most recent page; page_times keeps every page this round.
    time_on_page = models.IntegerField(initial=0)
    page_times   = models.LongStringField(initial='{}')
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
    # Computer opponent — set on the human player in rounds played against the computer
    vs_computer     = models.BooleanField(initial=False)
    computer_type   = models.StringField(blank=True)   # 'good' or 'bad'
    computer_guess  = models.StringField(blank=True)
    computer_payoff = models.FloatField(initial=0.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed_jar(session_code, match_number):
    return abs(hash((session_code, 'stage3_jar', match_number))) % (2 ** 31)

def _seed_draw(session_code, round_number):
    return abs(hash((session_code, 'stage3_draw', round_number))) % (2 ** 31)

def _seed_computer_type(session_code, group_id, match_number):
    return abs(hash((session_code, 'stage3_computer_type', group_id, match_number))) % (2 ** 31)


def _get_jar(session_code, match_number):
    """Jar for a given match: independently randomly assigned per match."""
    rng = np.random.default_rng(_seed_jar(session_code, match_number))
    return 'Red' if rng.integers(0, 2) == 0 else 'Blue'


def _shuffled_ball_order(session_code, salt, round_number, n_red, n_blue):
    """Deterministic per-round shuffle so red/blue balls display in mixed order."""
    colors = ['R'] * n_red + ['B'] * n_blue
    seed = abs(hash((session_code, salt, round_number, n_red, n_blue))) % (2 ** 31)
    np.random.default_rng(seed).shuffle(colors)
    return colors


def _get_draw(session_code, round_number):
    """Return (draw_red, draw_blue) for N_DRAWS with-replacement draws."""
    jar   = _get_jar(session_code, _match_number(round_number))
    rng   = np.random.default_rng(_seed_draw(session_code, round_number))
    n_red = C.RED_JAR_RED if jar == 'Red' else C.BLUE_JAR_RED
    k_red = int(rng.hypergeometric(n_red, C.N_BALLS - n_red, C.N_DRAWS))
    return k_red, C.N_DRAWS - k_red


def _match_number(round_number):
    return (round_number - 1) // C.ROUNDS_PER_MATCH + 1

def _round_in_match(round_number):
    return (round_number - 1) % C.ROUNDS_PER_MATCH + 1

def _is_bot_match(round_number):
    return _match_number(round_number) <= C.BOT_MATCHES

def _is_solo(player):
    """True for the leftover participant of an odd-sized session (no human partner)."""
    return len(player.group.get_players()) == 1

def _vs_computer(player):
    """True when this player's opponent this round is the computer."""
    return _is_bot_match(player.round_number) or _is_solo(player)

def _match_start_round(match_number):
    return (match_number - 1) * C.ROUNDS_PER_MATCH + 1

def _match_end_round(match_number):
    return match_number * C.ROUNDS_PER_MATCH


def _posterior(draw_red):
    """P(Red Jar | draw_red out of N_DRAWS balls), per-round."""
    engine = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine.update(n_draws=C.N_DRAWS, k_red=draw_red)
    return float(engine.belief[1])


def _computer_type(player):
    """
    Scheduled computer matches always use the bad bot. When the computer stands in for
    a missing human partner (solo participant in a human match), it is randomly good or bad.
    """
    if _is_bot_match(player.round_number):
        return 'bad'
    rng = np.random.default_rng(_seed_computer_type(
        player.session.code, player.group.id_in_subsession, _match_number(player.round_number),
    ))
    return 'good' if rng.integers(0, 2) == 0 else 'bad'


def _computer_guess(computer_type, draw_red, jar):
    """
    Good: guess the higher-posterior jar from this round's draw.
    Bad: always guess the wrong jar, so it never switches within a match.
    """
    if computer_type == 'bad':
        return 'Blue' if jar == 'Red' else 'Red'
    engine = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine.update_asymmetric(C.N_DRAWS, draw_red, lam_confirm=0.0, lam_disconfirm=0.0)
    return 'Red' if float(engine.belief[1]) > 0.5 else 'Blue'


def _assign_payoffs(p1, p2, jar):
    p1.is_correct = (p1.guess == jar)
    p2.is_correct = (p2.guess == jar)
    both_correct  = p1.is_correct and p2.is_correct
    both_wrong    = not p1.is_correct and not p2.is_correct
    if both_correct:
        p1.payoff_this_round = float(PAYOFF_BOTH_CORRECT)
        p2.payoff_this_round = float(PAYOFF_BOTH_CORRECT)
    elif both_wrong:
        p1.payoff_this_round = float(PAYOFF_BOTH_WRONG)
        p2.payoff_this_round = float(PAYOFF_BOTH_WRONG)
    else:
        p1.payoff_this_round = float(PAYOFF_ONE_CORRECT if p1.is_correct else PAYOFF_ONE_WRONG)
        p2.payoff_this_round = float(PAYOFF_ONE_CORRECT if p2.is_correct else PAYOFF_ONE_WRONG)


# ── Instructions widget ───────────────────────────────────────────────────────

def _instructions_vars():
    return dict(
        stage_instructions_bullets=[
            'Observe 10 balls drawn from a shared jar (the jar may change with each match).',
            'You and the other player each guess the jar independently.',
            'Payoffs depend on both guesses — see the payoff table.',
        ],
        show_payoff_table=True,
        payoff_both_correct=PAYOFF_BOTH_CORRECT,
        payoff_both_wrong=PAYOFF_BOTH_WRONG,
        payoff_one_correct=PAYOFF_ONE_CORRECT,
        payoff_one_wrong=PAYOFF_ONE_WRONG,
        abs_payoff_one_wrong=abs(PAYOFF_ONE_WRONG),
    )


# ── Session creation ──────────────────────────────────────────────────────────

def _pair_players(subsession):
    """
    Pair participants in order — [1, 2], [3, 4], ... — fixed for the whole stage.
    With an odd number, the last participant forms a group of one and plays the computer.
    """
    if subsession.round_number == 1:
        ids = [p.id_in_subsession for p in subsession.get_players()]
        subsession.set_group_matrix([ids[i:i + 2] for i in range(0, len(ids), 2)])
    else:
        subsession.group_like_round(1)


def creating_session(subsession):
    _pair_players(subsession)
    for player in subsession.get_players():
        player.jar_assignment = _get_jar(player.session.code, _match_number(subsession.round_number))


# ── Time-on-page tracking ─────────────────────────────────────────────────────

def _record_time_on_page(player, page_name):
    """Copy this page's time_on_page (seconds) into the per-round page_times JSON log."""
    times = json.loads(player.page_times)
    times[page_name] = player.time_on_page
    player.page_times = json.dumps(times)


# ── Pages ─────────────────────────────────────────────────────────────────────

class StageIntroPage(Page):
    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        return dict(
            **_instructions_vars(),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'StageIntroPage')


class MatchTransitionPage(Page):
    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def is_displayed(player):
        return player.round_number == 4

    @staticmethod
    def vars_for_template(player):
        match      = _match_number(player.round_number)
        prev_match = match - 1
        prev_start = _match_start_round(prev_match)
        prev_end   = _match_end_round(prev_match)
        prev_match_net = int(sum(
            player.in_round(r).payoff_this_round
            for r in range(prev_start, prev_end + 1)
        ))
        return dict(
            **_instructions_vars(),
            match_number=match,
            prev_match_number=prev_match,
            prev_match_net=prev_match_net,
            is_vs_human=not _is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'MatchTransitionPage')


class DrawPage(Page):
    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def vars_for_template(player):
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)
        return dict(
            **_instructions_vars(),
            draw_red=draw_red,
            draw_blue=draw_blue,
            ball_order=_shuffled_ball_order(player.session.code, 'stage3_draw_page', player.round_number, draw_red, draw_blue),
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'DrawPage')


class ChoicePage(Page):
    form_model  = 'player'
    form_fields = ['guess', 'time_on_page']

    @staticmethod
    def error_message(player, values):
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def vars_for_template(player):
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)
        return dict(
            **_instructions_vars(),
            draw_red=draw_red,
            draw_blue=draw_blue,
            ball_order=_shuffled_ball_order(player.session.code, 'stage3_choice_page', player.round_number, draw_red, draw_blue),
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'ChoicePage')
        # Rounds against the computer (scheduled computer match, or a solo participant)
        # skip ResultsWaitPage, so results are computed here as soon as the guess is in —
        # no waiting for anyone.
        if not _vs_computer(player):
            return

        jar = _get_jar(player.session.code, _match_number(player.round_number))
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)

        player.group.draw_red         = draw_red
        player.group.draw_blue        = draw_blue
        player.group.balls_drawn_json = json.dumps(['R'] * draw_red + ['B'] * draw_blue)

        computer_type = _computer_type(player)
        computer = SimpleNamespace(guess=_computer_guess(computer_type, draw_red, jar))
        _assign_payoffs(player, computer, jar)

        player.vs_computer     = True
        player.computer_type   = computer_type
        player.computer_guess  = computer.guess
        player.computer_payoff = computer.payoff_this_round

        post = _posterior(draw_red)
        player.posterior_red = post


        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.payoff_this_round


class ResultsWaitPage(WaitPage):
    @staticmethod
    def vars_for_template(player):
        return _instructions_vars()

    @staticmethod
    def is_displayed(player):
        # Computer rounds (bot match 1-3, or solo participant): skipped for all players.
        # Results are computed in ChoicePage.before_next_page instead,
        # because the computer has no browser and can never arrive here.
        # Human round (4-6): both players must arrive before results are shown.
        return not _vs_computer(player)

    @staticmethod
    def after_all_players_arrive(group):
        # Only called in the human round (4-6).
        p1  = group.get_player_by_id(1)
        p2  = group.get_player_by_id(2)
        jar = _get_jar(group.session.code, _match_number(group.round_number))
        draw_red, draw_blue = _get_draw(group.session.code, group.round_number)

        group.draw_red         = draw_red
        group.draw_blue        = draw_blue
        group.balls_drawn_json = json.dumps(['R'] * draw_red + ['B'] * draw_blue)

        if not p1.guess or not p2.guess:
            raise ValueError(
                f"Stage 3 round {group.round_number}: missing guess in human match "
                f"(p1.guess={p1.guess!r}, p2.guess={p2.guess!r})"
            )

        _assign_payoffs(p1, p2, jar)

        post = _posterior(draw_red)
        p1.posterior_red = post
        p2.posterior_red = post

        for p in [p1, p2]:
            prev = p.participant.vars.get('cumulative_earnings', 0)
            p.participant.vars['cumulative_earnings'] = prev + p.payoff_this_round


class ResultsPage(Page):
    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def vars_for_template(player):
        if player.vs_computer:
            partner_guess = player.computer_guess
        else:
            partner_guess = player.group.get_player_by_id(3 - player.id_in_group).guess
        match   = _match_number(player.round_number)
        rig     = _round_in_match(player.round_number)
        return dict(
            **_instructions_vars(),
            match_number=match,
            round_in_match=rig,
            is_bot_match=_is_bot_match(player.round_number),
            partner_guess=partner_guess,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'ResultsPage')


class MatchSummaryPage(Page):
    """Shown at the end of each match (rounds 3, 6): reveals this match's total payoff."""

    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def is_displayed(player):
        return _round_in_match(player.round_number) == C.ROUNDS_PER_MATCH

    @staticmethod
    def vars_for_template(player):
        match       = _match_number(player.round_number)
        match_start = _match_start_round(match)
        match_total = int(sum(
            player.in_round(r).payoff_this_round
            for r in range(match_start, player.round_number + 1)
        ))
        return dict(
            **_instructions_vars(),
            match_number=match,
            match_total=match_total,
            is_vs_human=not _is_bot_match(player.round_number),
            is_last_match=match == C.NUM_MATCHES,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'MatchSummaryPage')


page_sequence = [
    StageIntroPage,
    MatchTransitionPage,
    DrawPage,
    ChoicePage,
    ResultsWaitPage,
    ResultsPage,
    MatchSummaryPage,
]
