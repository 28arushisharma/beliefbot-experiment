from otree.api import *
import numpy as np
import json
from types import SimpleNamespace

from beliefbot import BeliefEngine, hypergeometric_pmf  # noqa: F401

doc = """
Stage 4 — Writer-Reader with random samples.

12 rounds: 4 matches × 3 sub-rounds each.
Two jars: jar 1 (rounds 1-6, matches 1-2), jar 2 (rounds 7-12, matches 3-4, opposite colour).

Player 1 = Writer: sees all 20 balls + 6-ball sample shown to Reader; knows true jar.
Player 2 = Reader: sees 6-ball sample only; may request up to 3 additional draws at no cost.

Matches 1-2 (rounds 1-6): every participant plays Writer vs their own computer Reader (bad bot).
Matches 3-4 (rounds 7-12): vs real human Reader.

Good bot reader: Bayesian (lam_confirm=0, lam_disconfirm=0)  → guesses higher-posterior jar.
Bad  bot reader: always guesses the wrong jar (opposite of the true jar) for the whole match.

Payoffs per sub-round:
  Both correct:            $30 / $30
  Both wrong:               $0 / $0
  You correct, other wrong: $10 for you, -$3 for them
  You wrong, other correct: -$3 for you, $10 for them

ResultsPage does not reveal the true jar colour.
"""

PAYOFF_BOTH_CORRECT         = 30
PAYOFF_BOTH_WRONG           = 0
PAYOFF_CORRECT_OTHER_WRONG  = 10
PAYOFF_WRONG_OTHER_CORRECT  = -3


class C(BaseConstants):
    NAME_IN_URL       = 'stage4'
    PLAYERS_PER_GROUP = None   # pairs formed in creating_session; allows an odd leftover
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

    PAYOFF_BOTH_CORRECT        = PAYOFF_BOTH_CORRECT
    PAYOFF_BOTH_WRONG          = PAYOFF_BOTH_WRONG
    PAYOFF_CORRECT_OTHER_WRONG = PAYOFF_CORRECT_OTHER_WRONG
    PAYOFF_WRONG_OTHER_CORRECT = PAYOFF_WRONG_OTHER_CORRECT


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Seconds between page load and submit (set by JS on every page).
    # time_on_page holds the most recent page; page_times keeps every page this round.
    time_on_page = models.IntegerField(initial=0)
    page_times   = models.LongStringField(initial='{}')
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
    # Computer opponent — set on the human Writer in rounds played against the computer
    vs_computer     = models.BooleanField(initial=False)
    computer_type   = models.StringField(blank=True)   # 'good' or 'bad'
    computer_guess  = models.StringField(blank=True)
    computer_payoff = models.FloatField(initial=0.0)
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
    """
    Writer this round. In computer rounds everyone plays Writer against their own
    computer Reader; in human matches Player 1 writes and Player 2 reads.
    """
    return _vs_computer(player) or player.id_in_group == 1

def _is_reader(player):
    return not _is_writer(player)

def _is_solo(player):
    """True for the leftover participant of an odd-sized session (no human partner)."""
    return len(player.group.get_players()) == 1

def _vs_computer(player):
    """True when this player's opponent this round is the computer."""
    return _is_bot_match(player.round_number) or _is_solo(player)

def _computer_type(player):
    """
    Scheduled computer matches always use the bad bot. When the computer stands in for
    a missing human partner (solo participant in a human match), it is randomly good or bad.
    """
    if _is_bot_match(player.round_number):
        return 'bad'
    return _get_bot_reader_type(
        player.session.code, player.group.id_in_subsession, _match_number(player.round_number),
    )

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


def _shuffled_ball_order(session_code, salt, round_number, n_red, n_blue):
    """Deterministic per-round shuffle so red/blue balls display in mixed order."""
    colors = ['R'] * n_red + ['B'] * n_blue
    seed = abs(hash((session_code, salt, round_number, n_red, n_blue))) % (2 ** 31)
    np.random.default_rng(seed).shuffle(colors)
    return colors


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
        p1.payoff_this_round = float(PAYOFF_CORRECT_OTHER_WRONG if p1.is_correct else PAYOFF_WRONG_OTHER_CORRECT)
        p2.payoff_this_round = float(PAYOFF_CORRECT_OTHER_WRONG if p2.is_correct else PAYOFF_WRONG_OTHER_CORRECT)


# ── Instructions widget ───────────────────────────────────────────────────────

def _instructions_vars():
    return dict(
        stage_instructions_bullets=[
            'Writer sees all 20 balls and the 6-ball sample sent to Reader.',
            'Reader sees the sample only; may request up to 3 extra draws (free).',
            'Payoffs depend on both players\' jar guesses — see the payoff table.',
        ],
        show_payoff_table=True,
        payoff_both_correct=PAYOFF_BOTH_CORRECT,
        payoff_both_wrong=PAYOFF_BOTH_WRONG,
        payoff_one_correct=PAYOFF_CORRECT_OTHER_WRONG,
        payoff_one_wrong=PAYOFF_WRONG_OTHER_CORRECT,
        abs_payoff_one_wrong=abs(PAYOFF_WRONG_OTHER_CORRECT),
    )


# ── Session creation ──────────────────────────────────────────────────────────

def _pair_players(subsession):
    """
    Pair participants in order — [1, 2], [3, 4], ... — fixed for the whole stage.
    With an odd number, the last participant forms a group of one: they are the Writer
    and play against a computer Reader in every round.
    """
    if subsession.round_number == 1:
        ids = [p.id_in_subsession for p in subsession.get_players()]
        subsession.set_group_matrix([ids[i:i + 2] for i in range(0, len(ids), 2)])
    else:
        subsession.group_like_round(1)


def creating_session(subsession):
    _pair_players(subsession)
    for player in subsession.get_players():
        player.jar_assignment = _get_jar(player.session.code, subsession.round_number)


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
        # Round 1 only; everyone is a Writer against the computer at round 1
        return player.round_number == 1 and _is_writer(player)

    @staticmethod
    def vars_for_template(player):
        return dict(
            **_instructions_vars(),
            abs_payoff_wrong_other_correct=abs(PAYOFF_WRONG_OTHER_CORRECT),
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
        return player.round_number in (4, 7, 10)

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
        jar_changes = (player.round_number == 7)  # jar flips at match 3
        return dict(
            **_instructions_vars(),
            match_number=match,
            prev_match_number=prev_match,
            prev_match_net=prev_match_net,
            is_vs_human=not _is_bot_match(player.round_number),
            is_writer=_is_writer(player),
            jar_changes=jar_changes,
            # First human match: Player 2 switches from Writer (vs computer) to Reader
            role_just_changed=player.round_number == C.BOT_MATCHES * C.ROUNDS_PER_MATCH + 1,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'MatchTransitionPage')


class WriterPage(Page):
    form_model  = 'player'
    form_fields = ['guess', 'time_on_page']

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
            **_instructions_vars(),
            jar=jar,
            all_ball_order=_shuffled_ball_order(player.session.code, 'stage4_writer_all', player.round_number, n_red, n_blue),
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            sample_ball_order=_shuffled_ball_order(player.session.code, 'stage4_writer_sample', player.round_number, k_red_s, k_blue_s),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'WriterPage')
        # Rounds against the computer (scheduled computer match, or a solo Writer):
        # ResultsWaitPage and ReaderWaitPage are both skipped, so the Writer (P1)
        # computes and stores all round results here — no waiting for anyone.
        if not (_vs_computer(player) and _is_writer(player)):
            return

        jar = _get_jar(player.session.code, player.round_number)
        sample, _ = _get_sample_and_remaining(player.session.code, player.round_number)
        k_red_s  = sample.count('R')
        computer_type = _computer_type(player)

        # Computer reader uses only the initial sample (0 additional draws)
        post_good, post_bad = _compute_posteriors(C.SAMPLE_SIZE, k_red_s)

        # Good computer: guess higher-posterior jar; bad computer: guess opposite
        if computer_type == 'good':
            bot_guess = 'Red' if post_good > 0.5 else 'Blue'
        else:
            # Bad computer always guesses the wrong jar, so it never switches within a match
            bot_guess = 'Blue' if jar == 'Red' else 'Red'
        computer = SimpleNamespace(guess=bot_guess)

        # Writer knows the true jar; posterior is certain
        player.sample_json        = json.dumps(sample)
        player.posterior_red_good = 1.0 if jar == 'Red' else 0.0
        player.posterior_red_bad  = 1.0 if jar == 'Red' else 0.0

        _assign_payoffs(player, computer, jar)

        player.vs_computer     = True
        player.computer_type   = computer_type
        player.computer_guess  = computer.guess
        player.computer_payoff = computer.payoff_this_round


        # Computer has no cumulative earnings; update this player only
        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.payoff_this_round


class ReaderWaitPage(WaitPage):
    """
    Reader (P2) waits here until Writer (P1) has submitted WriterPage.
    Skipped entirely in bot rounds (is_displayed=False for all players).
    P1 has is_displayed=False in human rounds and passes instantly, releasing P2.
    """
    @staticmethod
    def vars_for_template(player):
        return _instructions_vars()

    @staticmethod
    def is_displayed(player):
        # Human rounds only; Reader waits, Writer passes instantly
        return not _is_bot_match(player.round_number) and _is_reader(player)


class ReaderPage(Page):
    form_model  = 'player'
    form_fields = ['n_additional_draws', 'guess', 'time_on_page']

    @staticmethod
    def error_message(player, values):
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def is_displayed(player):
        # Shown to human Reader only (P2 in human matches)
        return _is_reader(player)

    @staticmethod
    def vars_for_template(player):
        sample, _ = _get_sample_and_remaining(player.session.code, player.round_number)
        k_red_s   = sample.count('R')
        k_blue_s  = sample.count('B')
        all_add   = _get_all_additional_draws(player.session.code, player.round_number)
        return dict(
            **_instructions_vars(),
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            sample_ball_order=_shuffled_ball_order(player.session.code, 'stage4_reader_sample', player.round_number, k_red_s, k_blue_s),
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
        _record_time_on_page(player, 'ReaderPage')
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
    def vars_for_template(player):
        return _instructions_vars()

    @staticmethod
    def is_displayed(player):
        # Computer rounds (bot matches 1-6, or solo Writer): skipped; results computed
        # in WriterPage.before_next_page. Human rounds (7-12): both players wait
        return not _vs_computer(player)

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
    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def vars_for_template(player):
        if player.vs_computer:
            partner_guess, partner_payoff = player.computer_guess, player.computer_payoff
            n_add = 0  # the computer reader never requests additional draws
        else:
            partner = player.group.get_player_by_id(3 - player.id_in_group)
            partner_guess, partner_payoff = partner.guess, partner.payoff_this_round
            n_add = player.group.get_player_by_id(2).n_additional_draws
        match       = _match_number(player.round_number)
        rig         = _round_in_match(player.round_number)
        sample, _   = _get_sample_and_remaining(player.session.code, player.round_number)
        k_red_s     = sample.count('R')
        k_blue_s    = sample.count('B')
        # Additional balls drawn by Reader (for Reader's own view)
        all_add     = _get_all_additional_draws(player.session.code, player.round_number)
        reader_add_balls = all_add[:n_add]
        reader_add_red  = [b for b in reader_add_balls if b == 'R']
        reader_add_blue = [b for b in reader_add_balls if b == 'B']
        return dict(
            **_instructions_vars(),
            match_number=match,
            round_in_match=rig,
            is_bot_match=_is_bot_match(player.round_number),
            is_writer=_is_writer(player),
            guess=player.guess,
            payoff_this_round=int(player.payoff_this_round),
            abs_payoff_this_round=int(abs(player.payoff_this_round)),
            partner_guess=partner_guess,
            partner_payoff_this_round=int(partner_payoff),
            abs_partner_payoff_this_round=int(abs(partner_payoff)),
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            sample_ball_order=_shuffled_ball_order(player.session.code, 'stage4_results_sample', player.round_number, k_red_s, k_blue_s),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            reader_n_additional_draws=n_add,
            reader_add_red_balls=list(range(len(reader_add_red))),
            reader_add_blue_balls=list(range(len(reader_add_blue))),
            reader_add_ball_order=_shuffled_ball_order(
                player.session.code, 'stage4_results_add', player.round_number,
                len(reader_add_red), len(reader_add_blue),
            ),
            is_last_round=player.round_number == C.NUM_ROUNDS,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'ResultsPage')


page_sequence = [
    StageIntroPage,
    MatchTransitionPage,
    WriterPage,
    ReaderWaitPage,
    ReaderPage,
    ResultsWaitPage,
    ResultsPage,
]
