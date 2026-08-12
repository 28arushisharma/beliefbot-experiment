from otree.api import *
import numpy as np
import json

from beliefbot import BeliefEngine, hypergeometric_pmf  # noqa: F401

doc = """
Stage 2 — Rationality (Sequential Draws).

6 rounds split into two jar groups of 3 rounds each.
Rounds 1-3: jar group 1 (Red or Blue, randomly assigned from session code).
Rounds 4-6: jar group 2 (always the opposite colour of group 1).

Each round resets to a full 20-ball jar. Participants see 6 initial balls
drawn without replacement, then may pay $2 per ball for up to 4 more
(10 balls total). The first additional ball is rigged to match the jar
majority colour; subsequent ones are truly random from the remaining jar.

Payoff: $5 correct, $0 wrong, minus $2 per additional ball requested.
Net payoff can be negative.
"""

PAYOFF_CORRECT = 5   # dollars
DRAW_COST      = 2   # dollars per additional ball

_WANT_CHOICES = [[True,  'Yes, draw one more ball ($2)'],
                 [False, 'No, I will decide now']]
_WANT_LABEL   = 'Would you like to see one more ball?'


class C(BaseConstants):
    NAME_IN_URL       = 'stage2'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS        = 6
    ROUNDS_PER_JAR    = 3   # rounds 1-3 = jar group 1, rounds 4-6 = jar group 2

    N_BALLS         = 20
    N_INITIAL_DRAWS = 6
    MAX_ADDITIONAL  = 4   # max extra balls per round (10 balls total)

    RED_JAR_RED   = 14
    RED_JAR_BLUE  = 6
    BLUE_JAR_RED  = 6
    BLUE_JAR_BLUE = 14

    PAYOFF_CORRECT = PAYOFF_CORRECT
    DRAW_COST      = DRAW_COST


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Jar
    jar_group      = models.IntegerField()   # 1 (rounds 1-3) or 2 (rounds 4-6)
    jar_assignment = models.StringField()    # 'Red' or 'Blue'

    # Initial 6-ball draw
    initial_balls           = models.LongStringField()
    initial_red             = models.IntegerField()
    initial_blue            = models.IntegerField()
    posterior_after_initial = models.FloatField()

    # Additional draw decisions (want_N = decision to purchase ball N, made before seeing it)
    want_additional_1 = models.BooleanField(
        choices=_WANT_CHOICES, label=_WANT_LABEL, widget=widgets.RadioSelect)
    want_additional_2 = models.BooleanField(
        choices=_WANT_CHOICES, label=_WANT_LABEL, widget=widgets.RadioSelect,
        blank=True, initial=False)
    want_additional_3 = models.BooleanField(
        choices=_WANT_CHOICES, label=_WANT_LABEL, widget=widgets.RadioSelect,
        blank=True, initial=False)
    want_additional_4 = models.BooleanField(
        choices=_WANT_CHOICES, label=_WANT_LABEL, widget=widgets.RadioSelect,
        blank=True, initial=False)

    # Additional ball results (add_ball_1 is always the rigged draw)
    add_ball_1 = models.StringField(blank=True)
    add_ball_2 = models.StringField(blank=True)
    add_ball_3 = models.StringField(blank=True)
    add_ball_4 = models.StringField(blank=True)

    # Posteriors after each draw (stored for data; not shown to participants)
    posterior_after_add_1 = models.FloatField(initial=0.0)
    posterior_after_add_2 = models.FloatField(initial=0.0)
    posterior_after_add_3 = models.FloatField(initial=0.0)
    posterior_after_add_4 = models.FloatField(initial=0.0)

    # Round summary
    n_additional_draws = models.IntegerField(initial=0)
    total_draw_cost    = models.FloatField(initial=0.0)
    all_balls_json     = models.LongStringField()

    # Final
    guess = models.StringField(
        choices=[['Red', 'Red Jar (14 red, 6 blue)'],
                 ['Blue', 'Blue Jar (14 blue, 6 red)']],
        label='Which jar do you think these balls came from?',
        widget=widgets.RadioSelect,
    )
    is_correct   = models.BooleanField()
    gross_payoff = models.FloatField()
    net_payoff   = models.FloatField()


# ── Helpers ────────────────────────────────────────────────────────────────

def _seed_jar(session_code):
    """Seed for jar group 1 assignment (shared across all rounds in a session)."""
    return abs(hash((session_code, 'stage2_jar'))) % (2 ** 31)


def _seed_draw(session_code, round_number):
    """Seed for ball draws in a specific round."""
    return abs(hash((session_code, 'stage2_draw', round_number))) % (2 ** 31)


def _jar_group(round_number):
    return 1 if round_number <= C.ROUNDS_PER_JAR else 2


def _get_jar(session_code, jar_group):
    """
    Jar group 1: randomly Red or Blue (seeded from session code).
    Jar group 2: always the opposite of group 1.
    """
    rng   = np.random.default_rng(_seed_jar(session_code))
    jar_1 = 'Red' if rng.integers(0, 2) == 0 else 'Blue'
    if jar_group == 1:
        return jar_1
    return 'Blue' if jar_1 == 'Red' else 'Red'


def _compute_all_draws(session_code, round_number):
    """
    Return (jar, initial_balls, additional_balls).

    Jar is determined by jar group (same within rounds 1-3 and 4-6 respectively).
    Draws are WITHOUT REPLACEMENT per round: the full 20-ball jar is shuffled
    independently each round. The first 6 balls are the initial draw.
    additional_balls[0] = rigged ball (majority colour).
    additional_balls[1-3] = truly random from the remaining jar.
    """
    jar    = _get_jar(session_code, _jar_group(round_number))
    rng    = np.random.default_rng(_seed_draw(session_code, round_number))
    n_red  = C.RED_JAR_RED  if jar == 'Red' else C.BLUE_JAR_RED
    n_blue = C.N_BALLS - n_red
    balls  = np.array(['R'] * n_red + ['B'] * n_blue)
    rng.shuffle(balls)

    initial   = balls[:C.N_INITIAL_DRAWS].tolist()
    remaining = balls[C.N_INITIAL_DRAWS:].tolist()

    majority   = 'R' if jar == 'Red' else 'B'
    rigged_idx = next(i for i, b in enumerate(remaining) if b == majority)
    rigged     = remaining.pop(rigged_idx)

    additional = [rigged] + remaining[:C.MAX_ADDITIONAL - 1]
    return jar, initial, additional


def _posterior(balls_list):
    """P(Red Jar | draws), treating all draws as a single batch from a full 20-ball jar."""
    if not balls_list:
        return 0.5
    engine = BeliefEngine(n_balls=C.N_BALLS, hypotheses=[C.BLUE_JAR_RED, C.RED_JAR_RED])
    engine.update(n_draws=len(balls_list), k_red=sum(1 for b in balls_list if b == 'R'))
    return float(engine.belief[1])


def _group_context(round_number):
    """Return (jar_group, round_in_group, group_start, group_end) for a round."""
    g           = _jar_group(round_number)
    group_start = 1 if g == 1 else C.ROUNDS_PER_JAR + 1
    group_end   = C.ROUNDS_PER_JAR if g == 1 else C.NUM_ROUNDS
    rig         = round_number if g == 1 else round_number - C.ROUNDS_PER_JAR
    return g, rig, group_start, group_end


def _adddraw_vars(player, draw_number):
    jar, initial, additional = _compute_all_draws(player.session.code, player.round_number)
    all_so_far = initial + additional[:draw_number]
    new_ball   = additional[draw_number - 1]
    n_red  = all_so_far.count('R')
    n_blue = all_so_far.count('B')
    g, rig, group_start, group_end = _group_context(player.round_number)
    return dict(
        add_draw_number=draw_number,
        new_ball_is_red=new_ball == 'R',
        new_ball_color='Red' if new_ball == 'R' else 'Blue',
        red_balls=list(range(n_red)),
        blue_balls=list(range(n_blue)),
        draw_red=n_red,
        draw_blue=n_blue,
        n_draws_total=len(all_so_far),
        is_last_additional=draw_number == C.MAX_ADDITIONAL,
        draw_cost=DRAW_COST,
        cost_so_far=DRAW_COST * draw_number,
        max_additional=C.MAX_ADDITIONAL,
        jar_group=g,
        round_in_group=rig,
        group_start=group_start,
        group_end=group_end,
        cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
    )


def _adddraw_store(player, draw_number):
    jar, initial, additional = _compute_all_draws(player.session.code, player.round_number)
    ball = additional[draw_number - 1]
    setattr(player, f'add_ball_{draw_number}', ball)
    player.n_additional_draws = draw_number
    player.total_draw_cost    = float(DRAW_COST * draw_number)
    all_so_far = initial + additional[:draw_number]
    setattr(player, f'posterior_after_add_{draw_number}', _posterior(all_so_far))


# ── Session creation ───────────────────────────────────────────────────────

def creating_session(subsession):
    for player in subsession.get_players():
        g = _jar_group(player.round_number)
        player.jar_group      = g
        player.jar_assignment = _get_jar(player.session.code, g)


# ── Pages ─────────────────────────────────────────────────────────────────

class StageIntroPage(Page):
    @staticmethod
    def is_displayed(player): return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        return dict(
            draw_cost=DRAW_COST,
            max_balls_total=C.N_INITIAL_DRAWS + C.MAX_ADDITIONAL,
            rounds_per_jar_plus_1=C.ROUNDS_PER_JAR + 1,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class JarChangePage(Page):
    """Transition shown at round 4: jar group 2 begins."""

    @staticmethod
    def is_displayed(player): return player.round_number == C.ROUNDS_PER_JAR + 1

    @staticmethod
    def vars_for_template(player):
        group1_net = int(sum(
            player.in_round(r).net_payoff
            for r in range(1, C.ROUNDS_PER_JAR + 1)
        ))
        return dict(
            group1_net=group1_net,
            abs_group1_net=abs(group1_net),
            group1_max_gross=PAYOFF_CORRECT * C.ROUNDS_PER_JAR,
            rounds_per_jar_plus_1=C.ROUNDS_PER_JAR + 1,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


class DrawPage(Page):
    """Show 6 initial balls; ask whether to buy an additional ball."""

    form_model  = 'player'
    form_fields = ['want_additional_1']

    @staticmethod
    def error_message(player, values):
        if values.get('want_additional_1') is None:
            return 'Please select an option before continuing.'

    @staticmethod
    def vars_for_template(player):
        jar, initial, _ = _compute_all_draws(player.session.code, player.round_number)
        g, rig, group_start, group_end = _group_context(player.round_number)
        return dict(
            draw_red=initial.count('R'),
            draw_blue=initial.count('B'),
            red_balls=list(range(initial.count('R'))),
            blue_balls=list(range(initial.count('B'))),
            draw_cost=DRAW_COST,
            max_additional=C.MAX_ADDITIONAL,
            jar_group=g,
            round_in_group=rig,
            group_start=group_start,
            group_end=group_end,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        jar, initial, _ = _compute_all_draws(player.session.code, player.round_number)
        player.initial_balls           = json.dumps(initial)
        player.initial_red             = initial.count('R')
        player.initial_blue            = initial.count('B')
        player.posterior_after_initial = _posterior(initial)


class AdditionalDrawPage1(Page):
    form_model  = 'player'
    form_fields = ['want_additional_2']

    @staticmethod
    def error_message(player, values):
        if values.get('want_additional_2') is None:
            return 'Please select an option before continuing.'

    @staticmethod
    def is_displayed(player): return player.want_additional_1

    @staticmethod
    def vars_for_template(player): return _adddraw_vars(player, 1)

    @staticmethod
    def before_next_page(player, timeout_happened): _adddraw_store(player, 1)


class AdditionalDrawPage2(Page):
    form_model  = 'player'
    form_fields = ['want_additional_3']

    @staticmethod
    def error_message(player, values):
        if values.get('want_additional_3') is None:
            return 'Please select an option before continuing.'

    @staticmethod
    def is_displayed(player): return player.want_additional_2

    @staticmethod
    def vars_for_template(player): return _adddraw_vars(player, 2)

    @staticmethod
    def before_next_page(player, timeout_happened): _adddraw_store(player, 2)


class AdditionalDrawPage3(Page):
    form_model  = 'player'
    form_fields = ['want_additional_4']

    @staticmethod
    def error_message(player, values):
        if values.get('want_additional_4') is None:
            return 'Please select an option before continuing.'

    @staticmethod
    def is_displayed(player): return player.want_additional_3

    @staticmethod
    def vars_for_template(player): return _adddraw_vars(player, 3)

    @staticmethod
    def before_next_page(player, timeout_happened): _adddraw_store(player, 3)


class AdditionalDrawPage4(Page):
    """Show the 4th (final) additional ball. No further purchase option."""

    @staticmethod
    def is_displayed(player): return player.want_additional_4

    @staticmethod
    def vars_for_template(player): return _adddraw_vars(player, 4)

    @staticmethod
    def before_next_page(player, timeout_happened): _adddraw_store(player, 4)


class ChoicePage(Page):
    form_model  = 'player'
    form_fields = ['guess']

    @staticmethod
    def error_message(player, values):
        if not values.get('guess'):
            return 'Please select a jar before continuing.'

    @staticmethod
    def vars_for_template(player):
        jar, initial, additional = _compute_all_draws(player.session.code, player.round_number)
        n         = player.n_additional_draws
        all_balls = initial + additional[:n]
        g, rig, group_start, group_end = _group_context(player.round_number)
        return dict(
            draw_red=all_balls.count('R'),
            draw_blue=all_balls.count('B'),
            red_balls=list(range(all_balls.count('R'))),
            blue_balls=list(range(all_balls.count('B'))),
            n_total=len(all_balls),
            n_additional=n,
            total_cost=DRAW_COST * n,
            jar_group=g,
            round_in_group=rig,
            group_start=group_start,
            group_end=group_end,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        jar, initial, additional = _compute_all_draws(player.session.code, player.round_number)
        n         = player.n_additional_draws
        all_balls = initial + additional[:n]
        player.all_balls_json = json.dumps(all_balls)
        player.is_correct     = (player.guess == player.jar_assignment)
        player.gross_payoff   = float(PAYOFF_CORRECT) if player.is_correct else 0.0
        player.net_payoff     = player.gross_payoff - player.total_draw_cost
        prev = player.participant.vars.get('cumulative_earnings', 0)
        player.participant.vars['cumulative_earnings'] = prev + player.net_payoff


class ResultsPage(Page):
    @staticmethod
    def vars_for_template(player):
        jar, initial, additional = _compute_all_draws(player.session.code, player.round_number)
        n         = player.n_additional_draws
        all_balls = initial + additional[:n]
        g, rig, group_start, group_end = _group_context(player.round_number)

        group_net   = int(sum(
            player.in_round(r).net_payoff
            for r in range(group_start, player.round_number + 1)
        ))
        overall_net = int(sum(
            player.in_round(r).net_payoff
            for r in range(1, player.round_number + 1)
        ))
        net = int(player.net_payoff)
        return dict(
            draw_red=all_balls.count('R'),
            draw_blue=all_balls.count('B'),
            red_balls=list(range(all_balls.count('R'))),
            blue_balls=list(range(all_balls.count('B'))),
            n_total=len(all_balls),
            jar_assignment=player.jar_assignment,
            jar_group=g,
            round_in_group=rig,
            group_start=group_start,
            group_end=group_end,
            guess=player.guess,
            is_correct=player.is_correct,
            n_additional=n,
            total_draw_cost=int(player.total_draw_cost),
            gross_payoff=int(player.gross_payoff),
            net_payoff=net,
            abs_net_payoff=abs(net),
            group_net=group_net,
            abs_group_net=abs(group_net),
            group_max_gross=PAYOFF_CORRECT * rig,
            overall_net=overall_net,
            abs_overall_net=abs(overall_net),
            overall_max_gross=PAYOFF_CORRECT * player.round_number,
            is_last_round=player.round_number == C.NUM_ROUNDS,
            cumulative_earnings=int(player.participant.vars.get('cumulative_earnings', 0)),
        )


page_sequence = [
    StageIntroPage,
    JarChangePage,
    DrawPage,
    AdditionalDrawPage1,
    AdditionalDrawPage2,
    AdditionalDrawPage3,
    AdditionalDrawPage4,
    ChoicePage,
    ResultsPage,
]
