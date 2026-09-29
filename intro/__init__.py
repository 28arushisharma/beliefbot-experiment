from otree.api import *

doc = """
General instructions shown once at the very start of the experiment, before Stage 1.
"""


class C(BaseConstants):
    NAME_IN_URL       = 'intro'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS        = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pass


# ── Pages ─────────────────────────────────────────────────────────────────────

class GeneralInstructionsPage(Page):
    pass


page_sequence = [GeneralInstructionsPage]
