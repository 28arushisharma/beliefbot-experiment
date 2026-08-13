# BeliefBot Experiment — Project Context

## Overview
This is a 6-stage economics experiment studying confirmation bias and information provision.
Built in oTree. Uses the `beliefbot` Python library for Bayesian belief updating.

## Jar Setup
- Red Jar: 14 red, 6 blue (always described this way, never percentages)
- Blue Jar: 14 blue, 6 red
- 20 balls total
- Jar resets to 20 balls at the start of every sub-round

## Round Structure (all stages)
- All stages use 2 jar groups × 3 sub-rounds = 6 rounds total (NUM_ROUNDS = 6, ROUNDS_PER_JAR = 3)
- Rounds 1-3 share the same jar (jar group 1, randomly Red or Blue from session code)
- Rounds 4-6 share a different jar (jar group 2, always opposite colour of group 1)
- A JarChangePage transition is shown at round 4 in every stage
- Within a jar group the colour is fixed; the jar resets to 20 balls before each sub-round

## Global Rules
- Always warn participants not to refresh the page
- Show a transition screen between stages: 'You are now entering Stage X'
- Track cumulative earnings across all stages, displayed at top of screen
- Placeholder payoff amounts use the variable PAYOFF_CORRECT = 5 (dollars)
- Store beliefbot posterior probabilities in data for every draw (not displayed to participants)
- Use `from beliefbot import BeliefEngine, hypergeometric_pmf` in app logic
- All HTML templates use Django template syntax only — no Jinja2 filters
- Never use |range, |items, |zip filters in templates
- Always use {% %} for tags, {{ }} for variables

## Stage 1 — Bayesian Learning
- Individual, no multiplayer
- 6 rounds: rounds 1-3 use jar group 1, rounds 4-6 use jar group 2
- Each round: 6 balls drawn WITH replacement from the full 20-ball jar
- JarChangePage shown at round 4
- Payoff: $5 correct, $0 wrong (placeholder)
- Store: jar_group, jar_assignment, balls drawn, guess, correct/wrong, payoff, posterior

## Stage 2 — Rationality (Sequential Draws)
- Individual, no multiplayer
- 6 rounds: rounds 1-3 use jar group 1, rounds 4-6 use jar group 2 (same 2×3 structure as Stage 1)
- JarChangePage shown at round 4
- Each round: 6 balls drawn WITHOUT replacement from a fresh 20-ball jar
- Participant may then purchase up to 4 additional balls at $2 each (10 balls total max)
- First additional ball is rigged to match jar majority colour (displayed as if random)
- Subsequent additional balls are truly random from remaining jar
- Net payoff = $5 if correct else $0, minus $2 per additional ball purchased (can be negative)
- Store: jar_group, jar_assignment, initial_balls, each additional ball, n_additional_draws, total_draw_cost, guess, correct/wrong, gross_payoff, net_payoff, posterior after each draw

## Stage 3 — Conformity (Coordination Game)
- PLAYERS_PER_GROUP = 2; NUM_ROUNDS = 12
- One jar for ALL 12 rounds, assigned once from session code (_seed_jar)
- 4 matches × 3 sub-rounds = 12 rounds total; 10 balls per sub-round, WITH replacement
- Matches 1-2 (rounds 1-6): vs computer bot; matches 3-4 (rounds 7-12): vs real human
- Bot is Player 2 (id_in_group==2); all their pages use is_displayed=False during bot matches
- Bot choice: fixed Red/Blue set once in participant.vars['bot_choice'] at round 1 creating_session
- Bot choice set in after_all_players_arrive (ResultsWaitPage) from participant.vars
- MatchTransitionPage shown at rounds 4, 7, 10 (non-bot players only)
- Payoffs: both correct=$30, both wrong=$20, one correct=$10/$0
- Store per sub-round: draw_red, draw_blue, balls_drawn_json (Group), guess, is_correct, payoff_this_round, posterior_red (Player)
- Cumulative earnings updated only for actively-playing players (not bot during bot matches)

## Stage 4 — Writer-Reader Game
- PLAYERS_PER_GROUP = 2; NUM_ROUNDS = 12
- Player 1 = Writer (sees all 20 balls + 6-ball sample); Player 2 = Reader (sees sample only)
- 4 matches × 3 sub-rounds = 12 rounds; 2 jars (jar 1 rounds 1-6, jar 2 rounds 7-12, opposite colours)
- Matches 1-2 (rounds 1-6): vs computer Reader (bot P2); matches 3-4 (rounds 7-12): vs human Reader
- Bot reader type ('good' or 'bad') assigned per match from session code; stored in participant.vars['bot_reader_types'] as {1: ..., 2: ...}
- Good bot: lam_confirm=0, lam_disconfirm=0 → guesses higher-posterior jar
- Bad bot: lam_confirm=0, lam_disconfirm=0.9 → guesses opposite of biased posterior mode
- Sample: 6 balls drawn WITHOUT replacement from jar; additional draws from remaining 14 (up to 3, no cost)
- Bot reader always uses 0 additional draws; bot guess computed in WriterPage.before_next_page
- ReaderWaitPage (WaitPage): Reader waits for Writer; skipped in bot rounds; P1 passes instantly in human rounds
- ResultsWaitPage (WaitPage): skipped for bot rounds; after_all_players_arrive for human rounds
- In bot rounds: all computation in WriterPage.before_next_page (same pattern as Stage 3)
- ReaderPage uses JS to reveal pre-computed additional balls one at a time; n_additional_draws hidden input
- Writer's posterior stored as 1.0/0.0 (knows true jar); Reader posteriors stored for good and bad engines
- Posteriors stored: posterior_red_good (lam_disconfirm=0), posterior_red_bad (lam_disconfirm=0.9)
- Cumulative earnings updated: Writer only in bot rounds; both players in human rounds
- ResultsPage shows Writer how many additional draws Reader requested

## Stages 5-6
- To be built after Stages 1-4 are complete
- See project notes for full design

## oTree Conventions Used
- PLAYERS_PER_GROUP = None for individual stages
- PLAYERS_PER_GROUP = 2 for multiplayer stages
- Bot pages use is_displayed = lambda player: False
- Bot state stored in participant.vars
- Draws computed deterministically from session code using numpy default_rng seeded from hash(session.code)
- Never store draw results in Group fields — compute from session code + round number instead (learned from Stage 0 demo)
