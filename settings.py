from os import environ

SESSION_CONFIGS = [
    dict(
        name='stage1',
        display_name='Stage 1 — Bayesian Learning',
        app_sequence=['stage1'],
        num_demo_participants=1,
    ),
    dict(
        name='stage2',
        display_name='Stage 2 — Sequential Information Acquisition',
        app_sequence=['stage2'],
        num_demo_participants=1,
    ),
    dict(
        name='stage3',
        display_name='Stage 3 — Conformity (Coordination Game)',
        app_sequence=['stage3'],
        num_demo_participants=2,  # Player 1 = human, Player 2 = bot (matches 1-2) / human (matches 3-4)
    ),
    dict(
        name='stage4',
        display_name='Stage 4 — Writer-Reader Game',
        app_sequence=['stage4'],
        num_demo_participants=2,  # Player 1 = Writer (human), Player 2 = bot Reader (matches 1-2) / human Reader (matches 3-4)
    ),
    dict(
        name='stage5',
        display_name='Stage 5 — Writer Chooses Sample',
        app_sequence=['stage5'],
        num_demo_participants=2,  # Player 1 = Writer (human), Player 2 = bot Reader (matches 1-2) / human Reader (matches 3-4)
    ),
    dict(
        name='jar_beliefs',
        display_name='Jar & Balls: Belief Updating',
        app_sequence=['jar_beliefs'],
        num_demo_participants=2,  # Player 1 = human, Player 2 = computational bot
    ),
]

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc=""
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '8778482555757'
