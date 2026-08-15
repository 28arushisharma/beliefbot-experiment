from otree.api import *

doc = """
Stage break — experimenter-controlled pause between Stage 3 and Stage 4.

Participants see a holding message and cannot advance themselves.
The experimenter advances everyone via the oTree admin panel
("Advance slowest participants").
"""


class C(BaseConstants):
    NAME_IN_URL       = 'stage_break'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS        = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pass


def _instructions_vars():
    return dict(
        stage_instructions_bullets=[
            'This is a scheduled break between Part 1 and Part 2 of the experiment.',
            'Please wait for further instructions from the experimenter.',
        ],
        show_payoff_table=False,
    )


class BreakPage(Page):
    @staticmethod
    def vars_for_template(player):
        return _instructions_vars()


page_sequence = [BreakPage]
