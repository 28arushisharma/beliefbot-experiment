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
- 2 rounds vs computer + 2 rounds vs real player (bot fills if mismatch)
- Same jar for all 4 rounds, told to participant
- 10 balls shown per round
- Computer picks Red or Blue randomly at start, sticks with same choice all 4 rounds
- Computer player handled by beliefbot BeliefEngine with lam_confirm=0, lam_disconfirm=0 (pure Bayesian) for jar choice
- Both players make independent choices, then see each other's choices
- Matched with same partner for all 4 rounds
- Payoffs: both correct = $30, both wrong = $20, one correct = $10 correct / $0 wrong
- Store: jar, draw, p1 choice, p2 choice (computer or human), both correct, payoffs

## Stages 4-6
- To be built after Stages 1-3 are complete
- Writer-reader design with sample selection
- See project notes for full design

## oTree Conventions Used
- PLAYERS_PER_GROUP = None for individual stages
- PLAYERS_PER_GROUP = 2 for multiplayer stages
- Bot pages use is_displayed = lambda player: False
- Bot state stored in participant.vars
- Draws computed deterministically from session code using numpy default_rng seeded from hash(session.code)
- Never store draw results in Group fields — compute from session code + round number instead (learned from Stage 0 demo)
