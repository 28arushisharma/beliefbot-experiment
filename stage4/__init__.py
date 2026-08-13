from otree.api import *
import numpy as np
import json

from beliefbot import BeliefEngine, hypergeometric_pmf  # noqa: F401

doc = """
Stage 4 — Writer-Reader with random samples.

12 rounds: 4 matches × 3 sub-rounds each.
Two jars: jar 1 (rounds 1-6, matches 1-2), jar 2 (rounds 7-12, matches 3-4, opposite colour).

Player 1 = Writer: sees all 20 balls + 6-ball sample shown to Reader; knows true jar.
Player 2 = Reader: sees 6-ball sample only; may request up to 3 additional draws at no cost.

Matches 1-2 (rounds 1-6): vs computer Reader (randomly 'good' or 'bad' per match).
Matches 3-4 (rounds 7-12): vs real human Reader.

Good bot reader: Bayesian (lam_confirm=0, lam_disconfirm=0)  → guesses higher-posterior jar.
Bad  bot reader: Biased  (lam_confirm=0, lam_disconfirm=0.9) → guesses opposite of posterior mode.

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
    NAME_IN_URL       = 'stage4'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS        = 12
    ROUNDS_PER_MATCH  = 3
    NUM_MATCHES       = 4
    BOT_MATCHES       = 2   # matches 1-2 vs bot reader; matches 3-4 vs human reader

    N_BALLS        = 20
    SAMPLE_SIZE    = 6
    MAX_ADDITIONAL = 3

    RED_JAR_RED   = 14
    RED_JAR_BLUE  = 6
    BLUE_JAR_RED  = 6
    BLUE_JAR_BLUE = 14

    PAYOFF_BOTH_CORRECT = PAYOFF_BOTH_CORRECT
    PAYOFF_BOTH_WRONG   = PAYOFF_BOTH_WRONG
    PAYOFF_ONE_CORRECT  = PAYOFF_ONE_CORRECT
    PAYOFF_ONE_WRONG    = PAYOFF_ONE_WRONG


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    jar_assignment = models.StringField()
    guess = models.StringField(
        choices=[['Red',  'Red Jar (14 red, 6 blue)'],
                 ['Blue', 'Blue Jar (14 blue, 6 red)']],
        label='Which jar do you think these balls came from?',
        widget=widgets.RadioSelect,
        blank=True,
    )
    is_correct         = models.BooleanField(initial=False)
    payoff_this_round  = models.FloatField(initial=0.0)
    posterior_red_good = models.FloatField(initial=0.5)  # lam_disconfirm=0
    posterior_red_bad  = models.FloatField(initial=0.5)  # lam_disconfirm=0.9
    # Reader-specific (written to P2; set to defaults for P1)
    n_additional_draws    = models.IntegerField(initial=0, min_value=0, max_value=3)
    additional_draws_json = models.LongStringField(blank=True)
    sample_json           = models.LongStringField(blank=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed_jar(session_code):
    return abs(hash((session_code, 'stage4_jar'))) % (2 ** 31)

def _seed_sample(session_code, round_number):
    return abs(hash((session_code, 'stage4_sample', round_number))) % (2 ** 31)

def _seed_additional(session_code, round_number):
    return abs(hash((session_code, 'stage4_add', round_number))) % (2 ** 31)

def _seed_bot_type(session_code, group_id, match_number):
    return abs(hash((session_code, 'stage4_bot_type', group_id, match_number))) % (2 ** 31)


def _get_jar(session_code, round_number):
    """Jar 1 for rounds 1-6 (matches 1-2); jar 2 (opposite) for rounds 7-12 (matches 3-4)."""
    rng = np.random.default_rng(_seed_jar(session_code))
    jar1 = 'Red' if rng.integers(0, 2) == 0 else 'Blue'
    if round_number <= 6:
        return jar1
    return 'Blue' if jar1 == 'Red' else 'Red'


def _get_sample_and_remaining(session_code, round_number):
    """
    Draw SAMPLE_SIZE balls without replacement from the full jar.
    Returns (sample, remaining) each as a list of 'R'/'B'.
    """
    jar   = _get_jar(session_code, round_number)
    n_red = C.RED_JAR_RED if jar == 'Red' else C.BLUE_JAR_RED
    full_jar = ['R'] * n_red + ['B'] * (C.N_BALLS - n_red)
    rng = np.random.default_rng(_seed_sample(session_code, round_number))
    idx = rng.choice(C.N_BALLS, size=C.SAMPLE_SIZE, replace=False)
    idx_set = set(idx.tolist())
    sample    = [full_jar[i] for i in range(C.N_BALLS) if i in idx_set]
    remaining = [full_jar[i] for i in range(C.N_BALLS) if i not in idx_set]
    return sample, remaining


def _get_all_additional_draws(session_code, round_number):
    """Pre-compute all MAX_ADDITIONAL balls from remaining 14, deterministically."""
    _, remaining = _get_sample_and_remaining(session_code, round_number)
    rng = np.random.default_rng(_seed_additional(session_code, round_number))
    idx = rng.choice(len(remaining), size=C.MAX_ADDITIONAL, replace=False)
    return [remaining[i] for i in sorted(idx.tolist())]


def _get_bot_reader_type(session_code, group_id, match_number):
    rng = np.random.default_rng(_seed_bot_type(session_code, group_id, match_number))
    return 'good' if rng.integers(0, 2) == 0 else 'bad'


def _match_number(round_number):
    return (round_number - 1) // C.ROUNDS_PER_MATCH + 1

def _round_in_match(round_number):
    return (round_number - 1) % C.ROUNDS_PER_MATCH + 1

def _is_bot_match(round_number):
    return _match_number(round_number) <= C.BOT_MATCHES

def _is_writer(player):
    return player.id_in_group == 1

def _is_reader(player):
    return player.id_in_group == 2

def _is_bot_reader(player):
    """P2 acts as bot reader in matches 1-2."""
    return _is_bot_match(player.round_number) and _is_reader(player)

def _match_start_round(match_number):
    return (match_number - 1) * C.ROUNDS_PER_MATCH + 1

def _match_end_round(match_number):
    return match_number * C.ROUNDS_PER_MATCH


def _compute_posteriors(n_draws, k_red):
    """
    Return (posterior_red_good, posterior_red_bad) given draw totals.
    Good: Bayesian (lam_disconfirm=0). Bad: confirmation-biased (lam_disconfirm=0.9).
    """
    engine_good = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine_good.update_asymmetric(n_draws, k_red, lam_confirm=0.0, lam_disconfirm=0.0)
    engine_bad = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine_bad.update_asymmetric(n_draws, k_red, lam_confirm=0.0, lam_disconfirm=0.9)
    return float(engine_good.belief[1]), float(engine_bad.belief[1])


def _assign_payoffs(p1, p2, jar):
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


# ── Session creation ──────────────────────────────────────────────────────────

def creating_session(subsession):
    for player in subsession.get_players():
        player.jar_assignment = _get_jar(player.session.code, subsession.round_number)
        # Store bot reader type in P2's participant.vars at round 1
        if subsession.round_number == 1 and player.id_in_group == 2:
            types = {
                m: _get_bot_reader_type(
                    subsession.session.code,
                    player.group.id_in_subsession,
                    m,
                )
                for m in range(1, C.BOT_MATCHES + 1)
            }
            player.participant.vars['bot_reader_types'] = types


# ── Pages ─────────────────────────────────────────────────────────────────────

class StageIntroPage(Page):
    @staticmethod
    def is_displayed(player):
        # Round 1 only; P2 is always a bot at round 1 so this shows only to Writer
        return player.round_number == 1 and _is_writer(player)

    @staticmethod
    def vars_for_template(player):
        return dict(
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class MatchTransitionPage(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number in (4, 7, 10) and not _is_bot_reader(player)

    @staticmethod
    def vars_for_template(player):
        match      = _match_number(player.round_number)
        prev_match = match - 1
        prev_start = _match_start_round(prev_match)
        prev_end   = _match_end_round(prev_match)
        # P2 (Reader) had no earnings during bot matches
        was_bot_prev = _is_bot_match(prev_start) and _is_reader(player)
        if was_bot_prev:
            prev_match_net = 0
        else:
            prev_match_net = int(sum(
                player.in_round(r).payoff_this_round
                for r in range(prev_start, prev_end + 1)
            ))
        jar_changes = (player.round_number == 7)  # jar flips at match 3
        return dict(
            match_number=match,
            prev_match_number=prev_match,
            prev_match_net=prev_match_net,
            is_vs_human=not _is_bot_match(player.round_number),
            is_writer=_is_writer(player),
            jar_changes=jar_changes,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class WriterPage(Page):
    form_model  = 'player'
    form_fields = ['guess']

    @staticmethod
    def error_message(player, values):
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def is_displayed(player):
        return _is_writer(player)

    @staticmethod
    def vars_for_template(player):
        jar    = _get_jar(player.session.code, player.round_number)
        n_red  = C.RED_JAR_RED  if jar == 'Red'  else C.BLUE_JAR_RED
        n_blue = C.N_BALLS - n_red
        sample, _ = _get_sample_and_remaining(player.session.code, player.round_number)
        k_red_s  = sample.count('R')
        k_blue_s = sample.count('B')
        return dict(
            jar=jar,
            all_red_balls=list(range(n_red)),
            all_blue_balls=list(range(n_blue)),
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        # Bot rounds: ResultsWaitPage and ReaderWaitPage are both skipped.
        # Writer (P1) computes and stores all round results here.
        if not (_is_bot_match(player.round_number) and _is_writer(player)):
            return

        p2  = player.group.get_player_by_id(2)
        jar = _get_jar(player.session.code, player.round_number)
        sample, _ = _get_sample_and_remaining(player.session.code, player.round_number)
        k_red_s  = sample.count('R')
        match_num = _match_number(player.round_number)
        bot_type  = p2.participant.vars.get('bot_reader_types', {}).get(match_num, 'good')

        # Bot reader uses only the initial sample (0 additional draws)
        post_good, post_bad = _compute_posteriors(C.SAMPLE_SIZE, k_red_s)

        # Good bot: guess higher-posterior jar; bad bot: guess opposite
        if bot_type == 'good':
            bot_guess = 'Red' if post_good > 0.5 else 'Blue'
        else:
            bot_guess = 'Blue' if post_bad > 0.5 else 'Red'

        p2.guess                  = bot_guess
        p2.n_additional_draws     = 0
        p2.additional_draws_json  = json.dumps([])
        p2.sample_json            = json.dumps(sample)
        p2.posterior_red_good     = post_good
        p2.posterior_red_bad      = post_bad

        # Writer knows the true jar; posterior is certain
        player.sample_json        = json.dumps(sample)
        player.posterior_red_good = 1.0 if jar == 'Red' else 0.0
        player.posterior_red_bad  = 1.0 if jar == 'Red' else 0.0

        _assign_payoffs(player, p2, jar)

        # Update only Writer's cumulative earnings (bot has none)
        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.payoff_this_round


class ReaderWaitPage(WaitPage):
    """
    Reader (P2) waits here until Writer (P1) has submitted WriterPage.
    Skipped entirely in bot rounds (is_displayed=False for all players).
    P1 has is_displayed=False in human rounds and passes instantly, releasing P2.
    """
    @staticmethod
    def is_displayed(player):
        # Human rounds only; Reader waits, Writer passes instantly
        return not _is_bot_match(player.round_number) and _is_reader(player)


class ReaderPage(Page):
    form_model  = 'player'
    form_fields = ['n_additional_draws', 'guess']

    @staticmethod
    def error_message(player, values):
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def is_displayed(player):
        # Shown to human Reader only (P2 in human matches)
        return not _is_bot_reader(player) and _is_reader(player)

    @staticmethod
    def vars_for_template(player):
        sample, _ = _get_sample_and_remaining(player.session.code, player.round_number)
        k_red_s   = sample.count('R')
        k_blue_s  = sample.count('B')
        all_add   = _get_all_additional_draws(player.session.code, player.round_number)
        return dict(
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            additional_ball_1=all_add[0],
            additional_ball_2=all_add[1],
            additional_ball_3=all_add[2],
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        sample, _ = _get_sample_and_remaining(player.session.code, player.round_number)
        player.sample_json = json.dumps(sample)
        n_add      = player.n_additional_draws
        all_add    = _get_all_additional_draws(player.session.code, player.round_number)
        additional = all_add[:n_add]
        player.additional_draws_json = json.dumps(additional)
        k_red_s   = sample.count('R')
        k_red_add = sum(1 for b in additional if b == 'R')
        post_good, post_bad = _compute_posteriors(C.SAMPLE_SIZE + n_add, k_red_s + k_red_add)
        player.posterior_red_good = post_good
        player.posterior_red_bad  = post_bad


class ResultsWaitPage(WaitPage):
    @staticmethod
    def is_displayed(player):
        # Bot rounds (1-6): skipped; results computed in WriterPage.before_next_page
        # Human rounds (7-12): both players wait
        return not _is_bot_match(player.round_number)

    @staticmethod
    def after_all_players_arrive(group):
        # Human rounds only
        p1  = group.get_player_by_id(1)  # Writer
        p2  = group.get_player_by_id(2)  # Reader
        jar = _get_jar(group.session.code, group.round_number)

        # Store sample on Writer; compute Writer's posterior (knows true jar)
        sample, _ = _get_sample_and_remaining(group.session.code, group.round_number)
        p1.sample_json        = json.dumps(sample)
        p1.posterior_red_good = 1.0 if jar == 'Red' else 0.0
        p1.posterior_red_bad  = 1.0 if jar == 'Red' else 0.0

        if not p1.guess or not p2.guess:
            raise ValueError(
                f"Stage 4 round {group.round_number}: missing guess "
                f"(p1.guess={p1.guess!r}, p2.guess={p2.guess!r})"
            )

        _assign_payoffs(p1, p2, jar)

        for p in [p1, p2]:
            prev = p.participant.vars.get('cumulative_earnings', 0)
            p.participant.vars['cumulative_earnings'] = prev + p.payoff_this_round


class ResultsPage(Page):
    @staticmethod
    def is_displayed(player):
        return not _is_bot_reader(player)

    @staticmethod
    def vars_for_template(player):
        partner     = player.group.get_player_by_id(3 - player.id_in_group)
        match       = _match_number(player.round_number)
        rig         = _round_in_match(player.round_number)
        sample, _   = _get_sample_and_remaining(player.session.code, player.round_number)
        k_red_s     = sample.count('R')
        k_blue_s    = sample.count('B')
        reader      = player.group.get_player_by_id(2)
        n_add       = reader.n_additional_draws
        # Additional balls drawn by Reader (for Reader's own view)
        all_add     = _get_all_additional_draws(player.session.code, player.round_number)
        reader_add_balls = all_add[:n_add]
        reader_add_red  = [b for b in reader_add_balls if b == 'R']
        reader_add_blue = [b for b in reader_add_balls if b == 'B']
        return dict(
            jar_assignment=player.jar_assignment,
            match_number=match,
            round_in_match=rig,
            is_bot_match=_is_bot_match(player.round_number),
            is_writer=_is_writer(player),
            guess=player.guess,
            is_correct=player.is_correct,
            payoff_this_round=int(player.payoff_this_round),
            partner_guess=partner.guess,
            partner_is_correct=partner.is_correct,
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            reader_n_additional_draws=n_add,
            reader_add_red_balls=list(range(len(reader_add_red))),
            reader_add_blue_balls=list(range(len(reader_add_blue))),
            is_last_round=player.round_number == C.NUM_ROUNDS,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


page_sequence = [
    StageIntroPage,
    MatchTransitionPage,
    WriterPage,
    ReaderWaitPage,
    ReaderPage,
    ResultsWaitPage,
    ResultsPage,
]
