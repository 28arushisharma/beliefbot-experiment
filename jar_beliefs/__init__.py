from otree.api import *
import numpy as np

from beliefbot import BeliefEngine, hypergeometric_pmf

doc = """
Jar-and-ball belief updating experiment.

A jar contains 20 balls (true composition: 12 red, 8 blue).
Each of 5 rounds, 3 balls are drawn without replacement.

Player 1 (human) sees the draw and reports an integer belief about how many red
balls are in the jar (0–20). Player 2 is a confirmation-biased computational bot
whose belief is updated server-side via BeliefEngine (lam_confirm=0, lam_disconfirm=0.9).

After all 5 rounds the human sees their total quadratic (Brier) score.
"""


class C(BaseConstants):
    NAME_IN_URL = 'jar_beliefs'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 5

    N_BALLS = 20
    N_RED_TRUE = 12
    N_BLUE_TRUE = N_BALLS - N_RED_TRUE
    DRAWS_PER_ROUND = 3
    TRUE_IDX = N_RED_TRUE       # index of true hypothesis in [0 … N_BALLS] list

    LAM_CONFIRM = 0.0
    LAM_DISCONFIRM = 0.9

    HUMAN_ID = 1
    BOT_ID = 2


# ── Models ─────────────────────────────────────────────────────────────────

class Subsession(BaseSubsession):
    pass   # draws are computed on demand via _get_draw; nothing to initialise here


class Group(BaseGroup):
    pass   # draw_red/draw_blue removed — never reliably persisted from creating_session
           # in oTree 6; use _get_draw(player.session.code, round_number) instead


class Player(BasePlayer):
    # Human fills this via form; bot's value is set server-side in the WaitPage.
    reported_belief = models.IntegerField(
        min=0,
        max=C.N_BALLS,
        label="How many red balls do you think are in the jar?",
    )
    # Quadratic (Brier) score for this round, set server-side for both players.
    qs_this_round = models.FloatField()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_draw(session_code: str, round_number: int) -> tuple[int, int]:
    """
    Return (draw_red, draw_blue) for the given round, computed deterministically
    from the session code.

    Seeds a single RNG from the session code and replays all rounds up to
    round_number so the without-replacement jar state is correct at each step.
    Calling this for round N always produces the same answer for a given session,
    regardless of which page or ORM context asks for it — no database write needed.
    """
    seed = abs(hash(session_code)) % (2 ** 31)
    rng = np.random.default_rng(seed)
    jar_red   = C.N_RED_TRUE
    jar_total = C.N_BALLS
    for r in range(1, round_number + 1):
        k = int(rng.hypergeometric(jar_red, jar_total - jar_red, C.DRAWS_PER_ROUND))
        jar_red   -= k
        jar_total -= C.DRAWS_PER_ROUND
        if r == round_number:
            return k, C.DRAWS_PER_ROUND - k


def _wor_likelihoods(prev_red: int, n_remaining: int, k_red: int) -> np.ndarray:
    """
    P(k_red | hypothesis h) for each h in [0 … N_BALLS], adjusted for balls
    already removed in prior rounds (without-replacement across rounds).

    Under hypothesis h, after removing prev_red red balls, the jar holds
    (h − prev_red) red balls in n_remaining total.
    """
    return np.array([
        hypergeometric_pmf(
            h - prev_red, n_remaining, C.DRAWS_PER_ROUND, k_red
        )
        for h in range(C.N_BALLS + 1)
    ])


def _bot_update(engine: BeliefEngine, session_code: str, round_number: int) -> BeliefEngine:
    """
    Apply one round of asymmetric confirmation-bias updating to the bot's engine.

    Uses WOR-adjusted likelihoods so that balls drawn in previous rounds are
    correctly accounted for.  Draws are retrieved via _get_draw, not from the
    Group model, so this works correctly regardless of ORM state.
    """
    prev_red = sum(
        _get_draw(session_code, r)[0]
        for r in range(1, round_number)
    )
    n_remaining = C.N_BALLS - C.DRAWS_PER_ROUND * (round_number - 1)
    k_red, _    = _get_draw(session_code, round_number)

    liks    = _wor_likelihoods(prev_red, n_remaining, k_red)
    current = engine.belief
    unnorm  = liks * current
    total   = unnorm.sum()
    bayes   = current.copy() if total == 0 else unnorm / total

    # Per-hypothesis asymmetric blend
    lams   = np.where(bayes >= current, C.LAM_CONFIRM, C.LAM_DISCONFIRM)
    biased = (1.0 - lams) * bayes + lams * current
    engine.belief = biased / biased.sum()
    return engine


def _restore_bot_engine(participant_vars: dict) -> BeliefEngine:
    """Reconstruct a BeliefEngine from serialised participant state."""
    engine = BeliefEngine(n_balls=C.N_BALLS)
    if 'bot_belief' in participant_vars:
        engine.belief = np.array(participant_vars['bot_belief'])
    return engine


def _human_qs(reported: int) -> float:
    """
    Quadratic score for a human integer report, treated as a Dirac-delta belief.

    QS = 1 if reported == N_RED_TRUE, else −1.
    """
    p = np.zeros(C.N_BALLS + 1)
    p[reported] = 1.0
    dummy = BeliefEngine(n_balls=C.N_BALLS)
    return dummy.quadratic_score(p, C.TRUE_IDX)


# ── Pages ───────────────────────────────────────────────────────────────────

class DrawPage(Page):
    """Show the round's draw result to the human player. Bot skips this page."""

    @staticmethod
    def is_displayed(player: Player):
        return player.id_in_group == C.HUMAN_ID

    @staticmethod
    def vars_for_template(player: Player):
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)
        balls_before = C.N_BALLS - C.DRAWS_PER_ROUND * (player.round_number - 1)
        return dict(
            draw_red     = draw_red,
            draw_blue    = draw_blue,
            red_balls    = list(range(draw_red)),
            blue_balls   = list(range(draw_blue)),
            balls_before = balls_before,
            balls_after  = balls_before - C.DRAWS_PER_ROUND,
        )


class BeliefPage(Page):
    """Human enters their integer belief. Bot skips this page."""

    form_model  = 'player'
    form_fields = ['reported_belief']

    @staticmethod
    def is_displayed(player: Player):
        return player.id_in_group == C.HUMAN_ID

    @staticmethod
    def vars_for_template(player: Player):
        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)
        return dict(
            draw_red  = draw_red,
            draw_blue = draw_blue,
        )


class ResultsWaitPage(WaitPage):
    """
    Sync point between human and bot.

    after_all_players_arrive fires when the human submits BeliefPage and the
    bot (which skips DrawPage and BeliefPage) has also arrived.  The bot's
    belief engine is updated here and both players' QS values are stored.
    """

    @staticmethod
    def after_all_players_arrive(group: Group):
        human = group.get_player_by_id(C.HUMAN_ID)
        bot   = group.get_player_by_id(C.BOT_ID)

        # ── Update bot belief ───────────────────────────────────────────
        engine = _restore_bot_engine(bot.participant.vars)
        engine = _bot_update(engine, group.session.code, group.round_number)
        bot.participant.vars['bot_belief'] = engine.belief.tolist()

        # Bot reports the posterior mode
        bot.reported_belief = engine.hypotheses[int(np.argmax(engine.belief))]

        # Bot's QS from the full belief distribution
        bot.qs_this_round = engine.quadratic_score(engine.belief, C.TRUE_IDX)

        # ── Score human report ──────────────────────────────────────────
        human.qs_this_round = _human_qs(human.reported_belief)


class ResultsPage(Page):
    """Show both players' beliefs and the human's score for this round. Bot skips."""

    @staticmethod
    def is_displayed(player: Player):
        return player.id_in_group == C.HUMAN_ID

    @staticmethod
    def vars_for_template(player: Player):
        group = player.group
        bot   = group.get_player_by_id(C.BOT_ID)

        human_running = sum(
            group.in_round(r).get_player_by_id(C.HUMAN_ID).qs_this_round
            for r in range(1, player.round_number + 1)
        )
        bot_running = sum(
            group.in_round(r).get_player_by_id(C.BOT_ID).qs_this_round
            for r in range(1, player.round_number + 1)
        )

        draw_red, draw_blue = _get_draw(player.session.code, player.round_number)
        return dict(
            draw_red        = draw_red,
            draw_blue       = draw_blue,
            human_belief    = player.reported_belief,
            bot_belief      = bot.reported_belief,
            human_qs        = round(player.qs_this_round),
            bot_qs          = round(bot.qs_this_round),
            human_running   = round(human_running),
            bot_running     = round(bot_running),
            true_n_red      = C.N_RED_TRUE,
            is_last_round   = player.round_number == C.NUM_ROUNDS,
        )


class FinalResultsPage(Page):
    """
    Show the human's total quadratic score across all 5 rounds and a
    round-by-round summary table comparing them to the bot.
    Displayed only after the final round.
    """

    @staticmethod
    def is_displayed(player: Player):
        return (
            player.id_in_group == C.HUMAN_ID
            and player.round_number == C.NUM_ROUNDS
        )

    @staticmethod
    def vars_for_template(player: Player):
        group = player.group
        rounds_data = []
        human_total = 0.0
        bot_total   = 0.0

        for r in range(1, C.NUM_ROUNDS + 1):
            g_r     = group.in_round(r)
            human_r = g_r.get_player_by_id(C.HUMAN_ID)
            bot_r   = g_r.get_player_by_id(C.BOT_ID)
            human_total += human_r.qs_this_round
            bot_total   += bot_r.qs_this_round
            d_red, d_blue = _get_draw(player.session.code, r)
            rounds_data.append(dict(
                round_num    = r,
                draw_red     = d_red,
                draw_blue    = d_blue,
                human_belief = human_r.reported_belief,
                human_qs     = round(human_r.qs_this_round, 3),
                bot_belief   = bot_r.reported_belief,
                bot_qs       = round(bot_r.qs_this_round, 4),
            ))

        return dict(
            rounds_data   = rounds_data,
            human_total   = round(human_total, 3),
            bot_total     = round(bot_total, 4),
            true_n_red    = C.N_RED_TRUE,
            lam_confirm   = C.LAM_CONFIRM,
            lam_disconfirm = C.LAM_DISCONFIRM,
        )


# ── Bot participant ─────────────────────────────────────────────────────────

class PlayerBot(Bot):
    """
    Simulates Player 2 (the computational bot) through the oTree bot framework.

    DrawPage, BeliefPage, ResultsPage, and FinalResultsPage are all skipped
    (is_displayed returns False for Player 2).  The bot's actual belief
    computation happens in ResultsWaitPage.after_all_players_arrive.
    """

    def play_round(self):
        # No pages are displayed to the bot; oTree auto-advances past all of them.
        pass


# ── Page sequence ────────────────────────────────────────────────────────────

page_sequence = [DrawPage, BeliefPage, ResultsWaitPage, ResultsPage, FinalResultsPage]
