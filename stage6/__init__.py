from otree.api import *
import numpy as np
import json

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
Bad  bot reader: Biased  (lam_confirm=0, lam_disconfirm=0.9) → guesses opposite of posterior mode.

Payoffs per sub-round:
  Both correct:  $30
  Both wrong:    $20
  One correct:   $10 for correct / $0 for wrong
  Reader's net payoff = base payoff − additional draw cost (all matches).
"""

PAYOFF_BOTH_CORRECT = 30
PAYOFF_BOTH_WRONG   = 20
PAYOFF_ONE_CORRECT  = 10
PAYOFF_ONE_WRONG    = 0
DRAW_COST_PER_BALL  = 2.0  # ALL matches (no free draws)


class C(BaseConstants):
    NAME_IN_URL       = 'stage6'
    PLAYERS_PER_GROUP = 2
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

    PAYOFF_BOTH_CORRECT = PAYOFF_BOTH_CORRECT
    PAYOFF_BOTH_WRONG   = PAYOFF_BOTH_WRONG
    PAYOFF_ONE_CORRECT  = PAYOFF_ONE_CORRECT
    PAYOFF_ONE_WRONG    = PAYOFF_ONE_WRONG


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
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
    Generate N_SAMPLES samples of SAMPLE_SIZE balls each, drawn WITH replacement from the jar.
    Returns list of lists of 'R'/'B'.
    """
    jar   = _get_jar(session_code, round_number)
    n_red = C.RED_JAR_RED if jar == 'Red' else C.BLUE_JAR_RED
    rng   = np.random.default_rng(_seed_samples(session_code, round_number))
    samples = []
    for _ in range(C.N_SAMPLES):
        draws = rng.integers(0, C.N_BALLS, size=C.SAMPLE_SIZE)
        balls = ['R' if d < n_red else 'B' for d in draws]
        samples.append(balls)
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


# ── Session creation ──────────────────────────────────────────────────────────

def creating_session(subsession):
    for player in subsession.get_players():
        player.jar_assignment = _get_jar(player.session.code, subsession.round_number)
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
        return player.round_number == 1 and _is_writer(player)

    @staticmethod
    def vars_for_template(player):
        return dict(
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
            draw_cost_display=int(C.DRAW_COST_PER_BALL),
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
        was_bot_prev = _is_bot_match(prev_start) and _is_reader(player)
        prev_match_net = 0 if was_bot_prev else int(sum(
            player.in_round(r).payoff_this_round
            for r in range(prev_start, prev_end + 1)
        ))
        jar_changes = (player.round_number == 7)
        return dict(
            match_number=match,
            prev_match_number=prev_match,
            prev_match_net=prev_match_net,
            is_vs_human=not _is_bot_match(player.round_number),
            is_writer=_is_writer(player),
            jar_changes=jar_changes,
            draw_cost_display=int(C.DRAW_COST_PER_BALL),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class WriterPage(Page):
    form_model  = 'player'
    form_fields = ['selected_sample_index', 'guess']

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
            samples_template.append(dict(
                index      = i,
                number     = i + 1,   # 1-based label; avoids |add:1 filter in template
                n_red      = kr,
                n_blue     = C.SAMPLE_SIZE - kr,
                red_balls  = list(range(kr)),
                blue_balls = list(range(C.SAMPLE_SIZE - kr)),
            ))

        return dict(
            samples_template=samples_template,
            match_number=_match_number(player.round_number),
            round_in_match=_round_in_match(player.round_number),
            is_bot_match=_is_bot_match(player.round_number),
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
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

        if not _is_bot_match(player.round_number):
            return

        # Bot round: compute bot reader result here
        p2        = player.group.get_player_by_id(2)
        match_num = _match_number(player.round_number)
        bot_type  = p2.participant.vars.get('bot_reader_types', {}).get(match_num, 'good')

        k_red_s = selected_sample.count('R')
        post_good, post_bad = _compute_posteriors(C.SAMPLE_SIZE, k_red_s)

        if bot_type == 'good':
            bot_guess = 'Red' if post_good > 0.5 else 'Blue'
        else:
            bot_guess = 'Blue' if post_bad > 0.5 else 'Red'

        p2.guess                 = bot_guess
        p2.n_additional_draws    = 0
        p2.additional_draws_json = json.dumps([])
        p2.additional_draw_cost  = 0.0  # bot takes 0 draws; cost = 0
        p2.selected_sample_json  = json.dumps(selected_sample)
        p2.posterior_red_good    = post_good
        p2.posterior_red_bad     = post_bad

        _assign_payoffs(player, p2, jar)
        # Bot reader has no cumulative earnings; update Writer only
        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.payoff_this_round


class ReaderWaitPage(WaitPage):
    """Reader waits for Writer to submit. Skipped in bot rounds."""
    @staticmethod
    def is_displayed(player):
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
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
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
    def is_displayed(player):
        return not _is_bot_match(player.round_number)

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

        _assign_payoffs(p1, p2, jar)
        p2.payoff_this_round -= p2.additional_draw_cost  # net payoff for Reader

        for p in [p1, p2]:
            prev = p.participant.vars.get('cumulative_earnings', 0)
            p.participant.vars['cumulative_earnings'] = prev + p.payoff_this_round


class ResultsPage(Page):
    @staticmethod
    def is_displayed(player):
        return not _is_bot_reader(player)

    @staticmethod
    def vars_for_template(player):
        writer   = player.group.get_player_by_id(1)
        reader   = player.group.get_player_by_id(2)
        match    = _match_number(player.round_number)
        rig      = _round_in_match(player.round_number)
        samples  = _get_candidate_samples(player.session.code, player.round_number)
        sel_idx  = writer.selected_sample_index
        sample   = samples[sel_idx] if 0 <= sel_idx < C.N_SAMPLES else samples[0]
        k_red_s  = sample.count('R')
        k_blue_s = sample.count('B')

        n_add     = reader.n_additional_draws
        all_add   = _get_all_additional_draws(player.session.code, player.round_number)
        add_balls = all_add[:n_add]
        add_red   = [b for b in add_balls if b == 'R']
        add_blue  = [b for b in add_balls if b == 'B']

        return dict(
            match_number=match,
            round_in_match=rig,
            is_writer=_is_writer(player),
            payoff_this_round=int(player.payoff_this_round),
            selected_sample_number=sel_idx + 1,  # 1-based; avoids |add:1 filter in template
            sample_red_balls=list(range(k_red_s)),
            sample_blue_balls=list(range(k_blue_s)),
            n_red_sample=k_red_s,
            n_blue_sample=k_blue_s,
            reader_n_additional_draws=n_add,
            reader_add_cost=int(reader.additional_draw_cost),
            reader_add_red_balls=list(range(len(add_red))),
            reader_add_blue_balls=list(range(len(add_blue))),
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
