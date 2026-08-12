# BeliefBot Experiment — Project Context

## Overview
This is a 6-stage economics experiment studying confirmation bias and information provision.
Built in oTree. Uses the `beliefbot` Python library for Bayesian belief updating.

## Jar Setup
- Red Jar: 14 red, 6 blue (always described this way, never percentages)
- Blue Jar: 14 blue, 6 red
- 20 balls total
- Draws are without replacement within a round; jar resets to 20 at start of each round

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
- Same jar for all rounds, told to participant upfront
- 2 rounds, 3 draws of 6 balls per round
- Jar resets to 20 each round
- Payoff: $5 correct, $0 wrong (placeholder)
- Store: jar assignment, each ball drawn, guess, correct/wrong, payoff, posterior

## Stage 2 — Rationality (Sequential Draws)
- Individual, no multiplayer  
- New jar each round, tell participant jar may change
- Start with 6 balls shown
- First additional draw: rigged 4 matching + 2 opposite color
- Remaining additional draws: random from remaining jar
- Max 4 additional draws (10 balls total maximum)
- Cost per additional draw: $2 placeholder (deducted from earnings)
- 2 rounds
- Store: jar, all balls drawn, how many additional draws requested, money spent, final guess, correct/wrong

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
