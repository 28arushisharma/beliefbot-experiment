from otree.api import *
import numpy as np
import json
from types import SimpleNamespace

from beliefbot import BeliefEngine, hypergeometric_pmf  # noqa: F401

doc = """
Stage 6 — Writer does NOT know the true jar; additional draws always cost $2.

12 rounds: 4 matches × 3 sub-rounds each.
Two jars: jar 1 (rounds 1-6, matches 1-2), jar 2 (rounds 7-12, matches 3-4, opposite colour).

Player 1 = Writer: sees 6 candidate samples ONLY (not the full jar); selects one to send to Reader;
          does NOT know which jar the balls came from.
Player 2 = Reader: sees only the Writer's chosen sample; may request up to 3 additional draws
          at $2 each in ALL matches (no free draws).

Matches 1-2 (rounds 1-6):  vs computer Reader (bot P2); additional draws cost $2 each.
Matches 3-4 (rounds 7-12): vs real human Reader; additional draws cost $2 each.

ResultsPage: only payoffs revealed — no jar colour, no partner guess, no correctness indication.

Bad computer writer strategy (stored for analysis): selects the sample whose Bayesian posterior
most strongly points toward the WRONG jar (requires true jar from _get_jar; not shown to Writer).

Good bot reader: Bayesian (lam_confirm=0, lam_disconfirm=0)  → guesses higher-posterior jar.
Bad  bot reader: always guesses the wrong jar (opposite of the true jar) for the whole match.

Payoffs per sub-round depend on who the Writer is matched with:
  vs computer (conformity):        both correct $30, both wrong $20, one correct $10 / $0
  vs human    (non-conformity):    both correct $30, both wrong $0,  you correct $10, you wrong −$3
  Reader's net payoff = base payoff − additional draw cost (all matches).

A FinalPage is shown at the very end of the stage with total cumulative earnings
and a thank-you message.
"""

PAYOFF_BOTH_CORRECT = 30
PAYOFF_ONE_CORRECT  = 10
# Computer matches: conformity payoffs
PAYOFF_BOTH_WRONG_COMPUTER = 20
PAYOFF_ONE_WRONG_COMPUTER  = 0
# Human matches: non-conformity payoffs, with a wrong-answer penalty as in Stage 4
PAYOFF_BOTH_WRONG_HUMAN = 0
PAYOFF_ONE_WRONG_HUMAN  = -3
DRAW_COST_PER_BALL  = 2.0  # ALL matches (no free draws)


class C(BaseConstants):
    NAME_IN_URL       = 'stage6'
    PLAYERS_PER_GROUP = None   # pairs formed in creating_session; allows an odd leftover
    NUM_ROUNDS        = 12
    ROUNDS_PER_MATCH  = 3
    NUM_MATCHES       = 4
    BOT_MATCHES       = 2   # matches 1-2 vs bot reader; matches 3-4 vs human reader

    N_BALLS        = 20
    SAMPLE_SIZE    = 6
    N_SAMPLES      = 6     # number of candidate samples shown to Writer
    MAX_ADDITIONAL = 3

    RED_JAR_RED   = 14
    RED_JAR_BLUE  = 6
    BLUE_JAR_RED  = 6
    BLUE_JAR_BLUE = 14

    DRAW_COST_PER_BALL  = DRAW_COST_PER_BALL



class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Seconds between page load and submit (set by JS on every page).
    # time_on_page holds the most recent page; page_times keeps every page this round.
    time_on_page = models.IntegerField(initial=0)
    page_times   = models.LongStringField(initial='{}')
    jar_assignment     = models.StringField()
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

    # Writer fields
    selected_sample_index   = models.IntegerField(initial=-1, blank=True)
    selected_sample_json    = models.LongStringField(blank=True)
    misleading_sample_index = models.IntegerField(initial=-1)
    all_samples_json        = models.LongStringField(blank=True)

    # Reader fields
    n_additional_draws    = models.IntegerField(initial=0, min_value=0, max_value=3)
    additional_draws_json = models.LongStringField(blank=True)
    additional_draw_cost  = models.FloatField(initial=0.0)


# ── Seeds ─────────────────────────────────────────────────────────────────────

def _seed_jar(session_code):
    return abs(hash((session_code, 'stage6_jar'))) % (2 ** 31)

def _seed_samples(session_code, round_number):
    return abs(hash((session_code, 'stage6_samples', round_number))) % (2 ** 31)

def _seed_additional(session_code, round_number):
    return abs(hash((session_code, 'stage6_add', round_number))) % (2 ** 31)

def _seed_bot_reader_type(session_code, group_id, match_number):
    return abs(hash((session_code, 'stage6_bot_reader', group_id, match_number))) % (2 ** 31)


# ── Jar / draw helpers ────────────────────────────────────────────────────────

def _get_jar(session_code, round_number):
    """Jar 1 for rounds 1-6 (matches 1-2); jar 2 (opposite) for rounds 7-12 (matches 3-4)."""
    rng  = np.random.default_rng(_seed_jar(session_code))
    jar1 = 'Red' if rng.integers(0, 2) == 0 else 'Blue'
    if round_number <= 6:
        return jar1
    return 'Blue' if jar1 == 'Red' else 'Red'


def _get_candidate_samples(session_code, round_number):
    """
    Generate N_SAMPLES samples of SAMPLE_SIZE balls each. Returns list of lists of 'R'/'B'.

    Two samples are always included, at random positions:
      - a 3-3 sample (3 red, 3 blue);
      - a misleading 4-2 sample (4 of the wrong jar's colour, 2 of the true jar's colour).
    The rest are drawn WITH replacement from the jar.
    """
    jar   = _get_jar(session_code, round_number)
    n_red = C.RED_JAR_RED if jar == 'Red' else C.BLUE_JAR_RED
    rng   = np.random.default_rng(_seed_samples(session_code, round_number))
    samples = []
    for _ in range(C.N_SAMPLES):
        draws = rng.integers(0, C.N_BALLS, size=C.SAMPLE_SIZE)
        balls = ['R' if d < n_red else 'B' for d in draws]
        samples.append(balls)

    wrong, true = ('B', 'R') if jar == 'Red' else ('R', 'B')
    fixed = [['R'] * 3 + ['B'] * 3, [wrong] * 4 + [true] * 2]
    positions = rng.choice(C.N_SAMPLES, size=len(fixed), replace=False)
    for pos, balls in zip(positions.tolist(), fixed):
        rng.shuffle(balls)
        samples[pos] = balls
    return samples


def _get_all_additional_draws(session_code, round_number):
    """Pre-compute MAX_ADDITIONAL balls from the full jar (without replacement), deterministically."""
    jar      = _get_jar(session_code, round_number)
    n_red    = C.RED_JAR_RED if jar == 'Red' else C.BLUE_JAR_RED
    full_jar = ['R'] * n_red + ['B'] * (C.N_BALLS - n_red)
    rng      = np.random.default_rng(_seed_additional(session_code, round_number))
    idx      = rng.choice(C.N_BALLS, size=C.MAX_ADDITIONAL, replace=False)
    return [full_jar[i] for i in sorted(idx.tolist())]


def _get_bot_reader_type(session_code, group_id, match_number):
    rng = np.random.default_rng(_seed_bot_reader_type(session_code, group_id, match_number))
    return 'good' if rng.integers(0, 2) == 0 else 'bad'


def _shuffled_ball_order(session_code, salt, round_number, n_red, n_blue):
    """Deterministic per-round shuffle so red/blue balls display in mixed order."""
    colors = ['R'] * n_red + ['B'] * n_blue
    seed = abs(hash((session_code, salt, round_number, n_red, n_blue))) % (2 ** 31)
    np.random.default_rng(seed).shuffle(colors)
    return colors


# ── Round / match helpers ─────────────────────────────────────────────────────

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
    return _is_bot_match(player.round_number) and _is_reader(player)

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


# ── Belief / posterior helpers ────────────────────────────────────────────────

def _compute_posteriors(n_draws, k_red):
    """Return (posterior_red_good, posterior_red_bad)."""
    engine_good = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine_good.update_asymmetric(n_draws, k_red, lam_confirm=0.0, lam_disconfirm=0.0)
    engine_bad  = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine_bad.update_asymmetric(n_draws, k_red, lam_confirm=0.0, lam_disconfirm=0.9)
    return float(engine_good.belief[1]), float(engine_bad.belief[1])


def _find_misleading_sample_index(samples, jar):
    """
    Bad computer writer strategy: return the index of the sample whose Bayesian
    posterior points most strongly toward the WRONG jar.
    Uses true jar for research purposes only (not shown to the human Writer in Stage 6).
    """
    wrong_is_blue = (jar == 'Red')
    best_idx, best_score = 0, -1.0
    for i, sample in enumerate(samples):
        k_red = sample.count('R')
        engine = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
        engine.update_asymmetric(C.SAMPLE_SIZE, k_red, lam_confirm=0.0, lam_disconfirm=0.0)
        post_red = float(engine.belief[1])
        score = (1.0 - post_red) if wrong_is_blue else post_red
        if score > best_score:
            best_score = score
            best_idx   = i
    return best_idx


def _payoff_table(vs_computer):
    """Base payoffs for one sub-round: conformity vs the computer, non-conformity vs a human."""
    return dict(
        both_correct=PAYOFF_BOTH_CORRECT,
        both_wrong=PAYOFF_BOTH_WRONG_COMPUTER if vs_computer else PAYOFF_BOTH_WRONG_HUMAN,
        one_correct=PAYOFF_ONE_CORRECT,
        one_wrong=PAYOFF_ONE_WRONG_COMPUTER if vs_computer else PAYOFF_ONE_WRONG_HUMAN,
    )


def _assign_payoffs(p1, p2, jar, vs_computer):
    table = _payoff_table(vs_computer)
    p1.is_correct = (p1.guess == jar)
    p2.is_correct = (p2.guess == jar)
    both_correct  = p1.is_correct and p2.is_correct
    both_wrong    = not p1.is_correct and not p2.is_correct
    if both_correct:
        p1.payoff_this_round = float(table['both_correct'])
        p2.payoff_this_round = float(table['both_correct'])
    elif both_wrong:
        p1.payoff_this_round = float(table['both_wrong'])
        p2.payoff_this_round = float(table['both_wrong'])
    else:
        p1.payoff_this_round = float(table['one_correct'] if p1.is_correct else table['one_wrong'])
        p2.payoff_this_round = float(table['one_correct'] if p2.is_correct else table['one_wrong'])


# ── Instructions widget ───────────────────────────────────────────────────────

def _payoff_vars(player):
    """This round's payoff table for templates (names match _base_instructions.html)."""
    table = _payoff_table(_vs_computer(player))
    return dict(
        payoff_both_correct=table['both_correct'],
        payoff_both_wrong=table['both_wrong'],
        payoff_one_correct=table['one_correct'],
        payoff_one_wrong=table['one_wrong'],
        abs_payoff_one_wrong=abs(table['one_wrong']),
    )


def _instructions_vars(player):
    return dict(
        stage_instructions_bullets=[
            'Writer does NOT know the jar — sees 6 candidate samples only.',
            'Writer chooses one sample to send; Reader may draw up to 3 extra balls ($2 each).',
            'Results reveal payoffs only — jar colour and the other player\'s guess are not shown.',
            'Payoffs depend on both players\' jar guesses — see the payoff table.',
        ],
        show_payoff_table=True,
        **_payoff_vars(player),
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
        return player.round_number == 1 and _is_writer(player)

    @staticmethod
    def vars_for_template(player):
        return dict(
            **_instructions_vars(player),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
            draw_cost_display=int(C.DRAW_COST_PER_BALL),
            payoff_table_heading='Payoffs per round',
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'StageIntroPage')


class MatchTransitionPage(Page):
    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def is_displayed(player):
        return player.round_number in (4, 7, 10) and not _is_bot_reader(player)

    @staticmethod
    def vars_for_template(player):
        match      = _match_number(player.round_number)
        prev_match = match - 1
        prev_start = _match_start_round(prev_match)
        prev_end   = _match_end_round(prev_match)
        was_bot_prev = _is_bot_match(prev_start) and _is_reader(player)
        prev_match_net = 0 if was_bot_prev else int(sum(
            player.in_round(r).payoff_this_round
            for r in range(prev_start, prev_end + 1)
        ))
        jar_changes = (player.round_number == 7)
        return dict(
            **_instructions_vars(player),
            match_number=match,
            prev_match_number=prev_match,
            prev_match_net=prev_match_net,
            abs_prev_match_net=abs(prev_match_net),
            is_vs_human=not _is_bot_match(player.round_number),
            is_writer=_is_writer(player),
            jar_changes=jar_changes,
            draw_cost_display=int(C.DRAW_COST_PER_BALL),
            payoff_table_heading='Payoffs for this match',
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'MatchTransitionPage')


class WriterPage(Page):
    form_model  = 'player'
    form_fields = ['selected_sample_index', 'guess', 'time_on_page']

    @staticmethod
    def error_message(player, values):
        idx = values.get('selected_sample_index')
        if idx is None or idx < 0:
            return 'Please click one of the 6 samples to choose what your reader will see.'
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def is_displayed(player):
        return _is_writer(player)

    @staticmethod
    def vars_for_template(player):
        samples = _get_candidate_samples(player.session.code, player.round_number)
        samples_template = []
        for i, sample in enumerate(samples):
            kr = sample.count('R')
            kb = C.SAMPLE_SIZE - kr
            samples_template.append(dict(
                index      = i,
                number     = i + 1,   # 1-based label; avoids |add:1 filter in template
                n_red      = kr,
                n_blue     = kb,
                ball_order = _shuffled_ball_order(player.session.code, f'stage6_writer_sample_{i}', player.round_number, kr, kb),
            ))

        return dict(
            **_instructions_vars(player),
            samples_template=samples_template,
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'WriterPage')
        if not _is_writer(player):
            return

        # True jar needed for misleading_sample_index research metric only
        jar     = _get_jar(player.session.code, player.round_number)
        samples = _get_candidate_samples(player.session.code, player.round_number)

        player.all_samples_json        = json.dumps(samples)
        player.misleading_sample_index = _find_misleading_sample_index(samples, jar)

        sel_idx = player.selected_sample_index
        if sel_idx < 0 or sel_idx >= C.N_SAMPLES:
            sel_idx = 0
        selected_sample = samples[sel_idx]
        player.selected_sample_json = json.dumps(selected_sample)

        # Writer's posteriors based on chosen sample (writer doesn't know true jar)
        k_red_w = selected_sample.count('R')
        player.posterior_red_good, player.posterior_red_bad = _compute_posteriors(C.SAMPLE_SIZE, k_red_w)

        if not _vs_computer(player):
            return

        # Computer round (scheduled computer match, or a solo Writer): compute the
        # computer reader's result here — no waiting for anyone
        computer_type = _computer_type(player)

        k_red_s = selected_sample.count('R')
        post_good, post_bad = _compute_posteriors(C.SAMPLE_SIZE, k_red_s)

        if computer_type == 'good':
            bot_guess = 'Red' if post_good > 0.5 else 'Blue'
        else:
            # Bad computer always guesses the wrong jar, so it never switches within a match
            bot_guess = 'Blue' if jar == 'Red' else 'Red'
        computer = SimpleNamespace(guess=bot_guess)

        _assign_payoffs(player, computer, jar, vs_computer=True)

        player.vs_computer     = True
        player.computer_type   = computer_type
        player.computer_guess  = computer.guess
        player.computer_payoff = computer.payoff_this_round

        # Scheduled computer match: P2's seat plays the computer — keep its record too
        if not _is_solo(player):
            p2 = player.group.get_player_by_id(2)
            p2.guess                 = computer.guess
            p2.is_correct            = computer.is_correct
            p2.payoff_this_round     = computer.payoff_this_round
            p2.n_additional_draws    = 0
            p2.additional_draws_json = json.dumps([])
            p2.additional_draw_cost  = 0.0  # computer takes 0 draws; cost = 0
            p2.selected_sample_json  = json.dumps(selected_sample)
            p2.posterior_red_good    = post_good
            p2.posterior_red_bad     = post_bad

        # Computer has no cumulative earnings; update Writer only
        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.payoff_this_round


class ReaderWaitPage(WaitPage):
    """Reader waits for Writer to submit. Skipped in bot rounds."""
    @staticmethod
    def vars_for_template(player):
        return _instructions_vars(player)

    @staticmethod
    def is_displayed(player):
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
        return not _is_bot_reader(player) and _is_reader(player)

    @staticmethod
    def vars_for_template(player):
        writer   = player.group.get_player_by_id(1)
        sel_idx  = writer.selected_sample_index
        samples  = _get_candidate_samples(player.session.code, player.round_number)
        sample   = samples[sel_idx] if 0 <= sel_idx < C.N_SAMPLES else samples[0]
        k_red_s  = sample.count('R')
        k_blue_s = sample.count('B')
        all_add  = _get_all_additional_draws(player.session.code, player.round_number)
        return dict(
            **_instructions_vars(player),
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            sample_ball_order=_shuffled_ball_order(player.session.code, 'stage6_reader_sample', player.round_number, k_red_s, k_blue_s),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            additional_ball_1=all_add[0],
            additional_ball_2=all_add[1],
            additional_ball_3=all_add[2],
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            draw_cost_per_ball=int(C.DRAW_COST_PER_BALL),  # always paid in stage 6
            is_paid_draws_js='true',                        # always; avoids |yesno in JS
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'ReaderPage')
        writer  = player.group.get_player_by_id(1)
        sel_idx = writer.selected_sample_index
        samples = _get_candidate_samples(player.session.code, player.round_number)
        sample  = samples[sel_idx] if 0 <= sel_idx < C.N_SAMPLES else samples[0]
        player.selected_sample_json = json.dumps(sample)

        n_add   = player.n_additional_draws
        all_add = _get_all_additional_draws(player.session.code, player.round_number)
        additional = all_add[:n_add]
        player.additional_draws_json = json.dumps(additional)

        k_red_s   = sample.count('R')
        k_red_add = sum(1 for b in additional if b == 'R')
        post_good, post_bad = _compute_posteriors(C.SAMPLE_SIZE + n_add, k_red_s + k_red_add)
        player.posterior_red_good = post_good
        player.posterior_red_bad  = post_bad

        # Draws always cost $2 in all matches
        player.additional_draw_cost = n_add * C.DRAW_COST_PER_BALL


class ResultsWaitPage(WaitPage):
    @staticmethod
    def vars_for_template(player):
        return _instructions_vars(player)

    @staticmethod
    def is_displayed(player):
        return not _vs_computer(player)

    @staticmethod
    def after_all_players_arrive(group):
        p1  = group.get_player_by_id(1)
        p2  = group.get_player_by_id(2)
        jar = _get_jar(group.session.code, group.round_number)

        if not p1.guess or not p2.guess:
            raise ValueError(
                f"Stage 6 round {group.round_number}: missing guess "
                f"(p1.guess={p1.guess!r}, p2.guess={p2.guess!r})"
            )

        _assign_payoffs(p1, p2, jar, vs_computer=False)
        p2.payoff_this_round -= p2.additional_draw_cost  # net payoff for Reader

        for p in [p1, p2]:
            prev = p.participant.vars.get('cumulative_earnings', 0)
            p.participant.vars['cumulative_earnings'] = prev + p.payoff_this_round


class ResultsPage(Page):
    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def is_displayed(player):
        return not _is_bot_reader(player)

    @staticmethod
    def vars_for_template(player):
        writer   = player.group.get_player_by_id(1)
        if player.vs_computer:
            n_add, add_cost = 0, 0.0  # the computer reader never requests additional draws
        else:
            reader = player.group.get_player_by_id(2)
            n_add, add_cost = reader.n_additional_draws, reader.additional_draw_cost
        match    = _match_number(player.round_number)
        rig      = _round_in_match(player.round_number)
        samples  = _get_candidate_samples(player.session.code, player.round_number)
        sel_idx  = writer.selected_sample_index
        sample   = samples[sel_idx] if 0 <= sel_idx < C.N_SAMPLES else samples[0]
        k_red_s  = sample.count('R')
        k_blue_s = sample.count('B')

        all_add   = _get_all_additional_draws(player.session.code, player.round_number)
        add_balls = all_add[:n_add]
        add_red   = [b for b in add_balls if b == 'R']
        add_blue  = [b for b in add_balls if b == 'B']

        return dict(
            **_instructions_vars(player),
            match_number=match,
            round_in_match=rig,
            is_writer=_is_writer(player),
            payoff_this_round=int(player.payoff_this_round),
            abs_payoff_this_round=int(abs(player.payoff_this_round)),
            selected_sample_number=sel_idx + 1,  # 1-based; avoids |add:1 filter in template
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            sample_ball_order=_shuffled_ball_order(player.session.code, 'stage6_results_sample', player.round_number, k_red_s, k_blue_s),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            reader_n_additional_draws=n_add,
            reader_add_cost=int(add_cost),
            reader_add_red_balls=list(range(len(add_red))),
            reader_add_blue_balls=list(range(len(add_blue))),
            reader_add_ball_order=_shuffled_ball_order(
                player.session.code, 'stage6_results_add', player.round_number,
                len(add_red), len(add_blue),
            ),
            is_last_round=player.round_number == C.NUM_ROUNDS,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'ResultsPage')


class FinalPage(Page):
    """Shown once at the very end of Stage 6 with total cumulative earnings."""

    form_model  = 'player'
    form_fields = ['time_on_page']

    @staticmethod
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player):
        return dict(
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        _record_time_on_page(player, 'FinalPage')


page_sequence = [
    StageIntroPage,
    MatchTransitionPage,
    WriterPage,
    ReaderWaitPage,
    ReaderPage,
    ResultsWaitPage,
    ResultsPage,
    FinalPage,
]
