"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import random
import time
import traceback
from itertools import combinations

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, get_table, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "the piggy 🐷"
AGENT_NAME = "FlushOverBoat"


# ===== YOUR CODE HERE =====
# Heads-up short-deck PLO5.
# Equity is Monte Carlo vs the opponent's range. The deal is a uniform prior.
# Each action they take reweights that prior by how often each hand class
# has taken that action this half (Bayes, with a Dirichlet prior). The
# weights are a few lookups on samples we already draw, so the 8-second
# bank still has room.

_TABLE = get_table()
_BITS = tuple(1 << c for c in range(36))
_PAIRS = tuple(combinations(range(5), 2))
_TRIPS = tuple(combinations(range(5), 3))
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# (board runouts, opponent hands) per street, then river samples and river
# refine samples. 0 falls behind, 1 is the opening sample. 2, 3, 4 and 5 are
# unlocked at the 200, 350, 500 and 750 hand checks, and only if the time
# left can pay for that sample until the end of the half.
_TIERS = (
    ({0: (5, 6), 3: (4, 6), 4: (4, 6)}, 60, 50),
    ({0: (8, 8), 3: (6, 8), 4: (5, 8)}, 90, 80),
    ({0: (12, 10), 3: (9, 10), 4: (8, 10)}, 135, 120),
    ({0: (13, 11), 3: (10, 11), 4: (9, 11)}, 152, 135),
    ({0: (14, 12), 3: (11, 12), 4: (10, 12)}, 170, 150),
    ({0: (15, 12), 3: (12, 12), 4: (10, 12)}, 190, 165),
)
# Seconds of choose_action time we allow per half. The judge's bank is 8.0
# and also pays for HTTP and JSON, so keep a wide margin.
_TIME_TARGET = 5.5
# (hand, tier, seconds of budget that must remain per hand still to come).
_STEPS = (
    (200, 2, 0.0050),
    (350, 3, 0.0060),
    (500, 4, 0.0072),
    (750, 5, 0.0088),
)
_NORMAL_AT = 0.0026

# Dirichlet priors: (check, call, raise). Each row sums to 40, so one half of
# showdowns can still move them. Rows are hand classes. Preflop classes are a
# cheap strength score. Postflop classes are the made hand: two pair or worse,
# straight, trips, full house, flush or better.
# Fitted to bots that bet by hand strength (equity thresholds and nits), then
# sharpened. A raise is usually trips or better; two pair checks. Maniacs are
# flattened later by _action_weights, so this prior is not used raw against them.
_PF_PRIOR = (
    (10.66, 25.42, 3.92),
    (6.43, 29.00, 4.57),
    (6.14, 26.61, 7.25),
    (0.42, 28.99, 10.59),
)
_POST_PRIOR = (
    (32.19, 5.00, 2.81),
    (17.94, 15.50, 6.56),
    (1.37, 5.74, 32.89),
    (0.87, 9.33, 29.81),
    (0.91, 5.19, 33.91),
)


def new_half(config: AgentConfig, memory: dict) -> None:
    """Start a half with priors for an unknown opponent."""
    memory["hands"] = 0
    memory["bets"] = 0
    memory["folds_to_bet"] = 0
    memory["opp_fold_hands"] = 0
    memory["opp_raises"] = 0
    memory["opp_passive"] = 0
    memory["bet_this_hand"] = False
    memory["hand_index"] = -1
    memory["hist_seen"] = 0
    memory["eq_key"] = None
    memory["eq"] = 0.5
    memory["eq_level"] = 0
    memory["pf"] = [list(row) for row in _PF_PRIOR]
    memory["post"] = [[list(row) for row in _POST_PRIOR] for _ in range(3)]
    memory["line"] = []
    memory["line_hand"] = -1
    memory["saw_button"] = False
    memory["weights"] = None
    memory["spent"] = 0.0
    memory["num_hands"] = max(1, int(config.num_hands))
    memory["tier"] = 1


def _mix(*nums: int) -> float:
    """Deterministic 0..1 mix from the cards, so a line is stable but not pure."""
    x = 2166136261
    for n in nums:
        x ^= n & 0xFFFFFFFF
        x = (x * 16777619) & 0xFFFFFFFF
    return x / 4294967296


def _best_with_board_masks(cards, board_masks) -> int:
    """Best Omaha score for 5 hole cards against one fully dealt board."""
    table = _TABLE
    cb = [_BITS[c] for c in cards]
    best = 0
    for i, j in _PAIRS:
        pair = cb[i] | cb[j]
        for bm in board_masks:
            score = table[pair | bm]
            if score > best:
                best = score
    return best


def _board_masks(board) -> list[int]:
    bb = [_BITS[c] for c in board]
    return [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPS]


def _pf_bucket(cards) -> int:
    """0..3 strength class from five hole cards. Higher is stronger."""
    rank_count = [0] * 9
    suit_count = [0] * 4
    for card in cards:
        rank_count[card >> 2] += 1
        suit_count[card & 3] += 1
    top = 0
    kept = 0
    pair = 0
    for rank in range(8, -1, -1):
        count = rank_count[rank]
        if count and kept < 3:
            top += rank
            kept += 1
        if count >= 2:
            pair += rank * (count - 1)
    suited = sum(1 for count in suit_count if count >= 2)
    score = top + 4 * pair + 3 * suited
    if score >= 28:
        return 3
    if score >= 22:
        return 2
    if score >= 16:
        return 1
    return 0


def _made_bucket(score: int) -> int:
    """0 two pair or worse, 1 straight, 2 trips, 3 full house, 4 flush+."""
    category = score >> 20
    if category <= 2:
        return 0
    if category == 3:
        return 1
    if category == 4:
        return 2
    if category == 5:
        return 3
    return 4


def _masks_for(board) -> list[int]:
    """Board-card masks for an Omaha hand on a 3, 4, or 5 card board."""
    bits = _BITS
    if len(board) == 3:
        return [bits[board[0]] | bits[board[1]] | bits[board[2]]]
    bb = [bits[c] for c in board]
    if len(board) == 4:
        return [
            bb[0] | bb[1] | bb[2],
            bb[0] | bb[1] | bb[3],
            bb[0] | bb[2] | bb[3],
            bb[1] | bb[2] | bb[3],
        ]
    return _board_masks(board)


def _partial_score(hole, board) -> int:
    """Best Omaha score using the cards currently on the board."""
    return _best_with_board_masks(hole, _masks_for(board))


def _equity(hole, board, seed: int, nrun: int, nopp: int, weights=None) -> float:
    """Showdown equity of `hole`. `weights` reweights opponent classes."""
    bits = _BITS
    used = 0
    for c in hole:
        used |= bits[c]
    for c in board:
        used |= bits[c]
    deck = [c for c in range(36) if (used & bits[c]) == 0]
    need = 5 - len(board)
    cur_masks = _masks_for(board) if weights is not None and board else None
    rng = random.Random(seed)
    won = 0.0
    seen = 0.0
    for _ in range(nrun):
        rng.shuffle(deck)
        full = tuple(board) + tuple(deck[:need])
        masks = _board_masks(full)
        mine = _best_with_board_masks(hole, masks)
        rest = deck[need:]
        for _opp in range(nopp):
            opp = rng.sample(rest, 5)
            theirs = _best_with_board_masks(opp, masks)
            if weights is None:
                weight = 1.0
            elif cur_masks is None:
                weight = weights[_pf_bucket(opp)]
            else:
                weight = weights[_made_bucket(_best_with_board_masks(opp, cur_masks))]
            seen += weight
            if mine > theirs:
                won += weight
            elif mine == theirs:
                won += 0.5 * weight
    if seen <= 0.0:
        return 0.5
    return won / seen


def _river_equity(hole, board, seed: int, samples: int, weights=None) -> float:
    """River equity. Every opponent pair is scored once, then hands are sampled."""
    table = _TABLE
    bits = _BITS
    used = 0
    for c in hole:
        used |= bits[c]
    for c in board:
        used |= bits[c]
    deck = [c for c in range(36) if (used & bits[c]) == 0]
    masks = _board_masks(board)
    pair_score = {}
    for i, a in enumerate(deck):
        ba = bits[a]
        for b in deck[i + 1:]:
            mask = ba | bits[b]
            best = 0
            for bm in masks:
                score = table[mask | bm]
                if score > best:
                    best = score
            pair_score[(a, b) if a < b else (b, a)] = best
    mine = _best_with_board_masks(hole, masks)
    rng = random.Random(seed)
    won = 0.0
    seen = 0.0
    for _ in range(samples):
        opp = rng.sample(deck, 5)
        opp.sort()
        best = 0
        for a, b in combinations(opp, 2):
            score = pair_score[(a, b)]
            if score > best:
                best = score
        weight = 1.0 if weights is None else weights[_made_bucket(best)]
        seen += weight
        if mine > best:
            won += weight
        elif mine == best:
            won += 0.5 * weight
    if seen <= 0.0:
        return 0.5
    return won / seen


def _opp_kind(obs: Observation) -> int | None:
    """Opponent's action this street: 0 check, 1 call, 2 raise, or None."""
    kind = None
    street = obs.street
    for rec in obs.action_history:
        if rec.street != street or rec.is_me:
            continue
        if rec.action in _RAISES:
            kind = 2
        elif rec.amount > 0:
            kind = 1
        elif kind is None:
            kind = 0
    return kind


def _action_weights(memory: dict, street: int, kind: int) -> tuple[float, ...]:
    """P(this action | hand class). A maniac's raises are pulled back toward even."""
    rows = memory["pf"] if street == 0 else memory["post"][street - 1]
    weights = []
    for row in rows:
        weights.append(row[kind] / (row[0] + row[1] + row[2]))
    aggro = _aggro(memory)
    # Flatten only for opponents who raise and do not fold. A preflop raiser
    # who then gives up is not bluffing the later streets.
    if kind == 2 and aggro > 0.60 and _fold_to_bet(memory) < 0.32:
        pull = min(0.85, (aggro - 0.50) * 1.4)
        mean = sum(weights) / len(weights)
        weights = [(1.0 - pull) * weight + pull * mean for weight in weights]
    # A player who folds to bets is not turning two pair into a bluff.
    # Downweight that class when they raise, which is the Red Chip question:
    # how often do they actually pull the trigger with a bluff candidate?
    if kind == 2 and memory["bets"] >= 10:
        folded = _fold_to_bet(memory)
        if folded > 0.40:
            scale = max(0.20, 1.15 - folded * 1.3)
            weights = list(weights)
            weights[0] *= scale
    return tuple(weights)


def _raw_equity(obs: Observation, memory: dict) -> float:
    """Equity for this street, weighted by the action they already took."""
    kind = _opp_kind(obs)
    key = (obs.hand_index, int(obs.street), obs.hole_cards, obs.board, kind)
    weights = None if kind is None else _action_weights(memory, int(obs.street), kind)
    memory["weights"] = weights
    if memory["eq_key"] == key:
        return memory["eq"]
    seed = (obs.hand_index + 1) * 10007 + (int(obs.street) + 3) * 97
    for c in obs.hole_cards:
        seed = seed * 131 + int(c) + 1
    for c in obs.board:
        seed = seed * 131 + int(c) + 19
    seed &= 0xFFFFFFFF
    budget, river_n, _ = _TIERS[memory["tier"]]
    n_board = len(obs.board)
    if n_board == 5:
        eq = _river_equity(obs.hole_cards, obs.board, seed, river_n, weights)
    else:
        nrun, nopp = budget[n_board]
        eq = _equity(obs.hole_cards, obs.board, seed, nrun, nopp, weights)
    memory["eq_key"] = key
    memory["eq"] = eq
    memory["eq_level"] = 1
    return eq


def _refine_raw(obs: Observation, memory: dict, raw: float, odds: float) -> float:
    """More samples when a big call is close. Same range weights as the first pass."""
    if memory["eq_level"] >= 2 or obs.to_call < 8 or abs(raw - odds) > 0.07:
        return raw
    weights = memory["weights"]
    seed = (obs.hand_index + 9) * 333 + int(obs.street) * 17 + 12345
    budget, _, refine_n = _TIERS[memory["tier"]]
    n_board = len(obs.board)
    if n_board == 5:
        extra = _river_equity(obs.hole_cards, obs.board, seed, refine_n, weights)
    else:
        nrun, nopp = budget[n_board]
        extra = _equity(obs.hole_cards, obs.board, seed, nrun, nopp, weights)
    mixed = 0.5 * (raw + extra)
    memory["eq"] = mixed
    memory["eq_level"] = 2
    return mixed


def _note_actions(obs: Observation, memory: dict) -> None:
    """Count opponent actions and keep this hand's line for the showdown update."""
    if memory["hand_index"] != obs.hand_index:
        memory["hand_index"] = obs.hand_index
        memory["hist_seen"] = 0
        memory["line"] = []
        memory["line_hand"] = obs.hand_index
    memory["saw_button"] = obs.is_button
    hist = obs.action_history
    line = memory["line"]
    for rec in hist[memory["hist_seen"]:]:
        line.append((int(rec.street), bool(rec.is_me), int(rec.action), int(rec.amount)))
        if not rec.is_me:
            if rec.action in _RAISES:
                memory["opp_raises"] += 1
            else:
                memory["opp_passive"] += 1
    memory["hist_seen"] = len(hist)


def _aggro(memory: dict) -> float:
    """Share of opponent actions that were raises. Prior is a normal player."""
    raises = memory["opp_raises"] + 4
    passive = memory["opp_passive"] + 8
    return raises / (raises + passive)


def _fold_to_bet(memory: dict) -> float:
    """How often a hand we bet ends with them folding. Prior is moderate."""
    return (memory["folds_to_bet"] + 2) / (memory["bets"] + 10)


def _hand_fold_rate(memory: dict) -> float:
    """How often they give up the hand without showdown."""
    return (memory["opp_fold_hands"] + 2) / (memory["hands"] + 8)


def _pot_odds(obs: Observation) -> float:
    call = min(obs.to_call, obs.my_stack)
    if call <= 0:
        return 0.0
    return call / (obs.pot + call)


def _clean_raises(obs: Observation) -> list[Action]:
    """Raises that do not put in chips the opponent cannot match.

    The engine does not give uncalled chips back, so betting more than the
    opponent's stack loses those chips whenever we do not win the pot.
    """
    cap = obs.opp_bet + obs.opp_stack
    clean = []
    for action in (Action.RAISE_POT, Action.RAISE_HALF):
        if action in obs.legal_actions and obs.raise_to.get(action, cap + 1) <= cap:
            clean.append(action)
    return clean


def _prefer(obs: Observation, ranked: list[Action]) -> Action:
    """First wanted action that is legal, otherwise check/call, otherwise fold."""
    legal = obs.legal_actions
    for action in ranked:
        if action in legal:
            return action
    if Action.CHECK_CALL in legal:
        return Action.CHECK_CALL
    return Action.FOLD


def _mark_bet(memory: dict, action: Action) -> Action:
    if action in _RAISES:
        memory["bet_this_hand"] = True
    return action


def _preflop(obs: Observation, eq: float, memory: dict) -> Action:
    aggro = _aggro(memory)
    opp_raises = sum(1 for rec in obs.action_history if (not rec.is_me) and rec.action in _RAISES)
    odds = _pot_odds(obs)
    clean = _clean_raises(obs)
    fold_rate = _hand_fold_rate(memory)
    steal = fold_rate > 0.40 and memory["hands"] >= 24

    if opp_raises == 0:
        pot_at = 0.46 if steal else 0.53
        half_at = 0.43 if steal else 0.47
        if eq >= pot_at and Action.RAISE_POT in clean:
            return _mark_bet(memory, Action.RAISE_POT)
        if eq >= half_at and Action.RAISE_HALF in clean:
            return _mark_bet(memory, Action.RAISE_HALF)
        # Opening the button costs 1 chip into a pot of 3. Almost every
        # five-card hand clears that price; fold only when the price is
        # larger and the hand is far behind.
        if obs.to_call == 0 or eq + 0.02 >= odds or eq >= 0.40:
            return Action.CHECK_CALL
        return _prefer(obs, [Action.FOLD])

    # A player who raises every hand has a random range, so get the money in
    # with anything clearly ahead. A quieter player's raise is stronger.
    hot = aggro > 0.70 and memory["hands"] >= 16
    pot_3b = 0.56 if hot else 0.63
    half_3b = 0.52 if hot else 0.57
    if eq >= pot_3b and Action.RAISE_POT in clean and obs.raises_this_street < 3:
        return _mark_bet(memory, Action.RAISE_POT)
    if eq >= half_3b and Action.RAISE_HALF in clean and obs.raises_this_street < 3:
        return _mark_bet(memory, Action.RAISE_HALF)
    need = odds - 0.01 if hot else odds + (0.0 if aggro > 0.60 else 0.035)
    if eq >= need or (obs.to_call <= obs.big_blind and eq >= 0.42):
        return Action.CHECK_CALL
    return _prefer(obs, [Action.FOLD])


def _postflop(obs: Observation, raw: float, memory: dict) -> Action:
    odds = _pot_odds(obs)
    eq = _refine_raw(obs, memory, raw, odds) if obs.to_call > 0 else raw
    aggro = _aggro(memory)
    f2b = _fold_to_bet(memory)
    ip = obs.is_button
    spr = obs.my_stack / max(1, obs.pot)
    clean = _clean_raises(obs)
    station = memory["bets"] >= 18 and f2b < 0.18
    maniac = aggro > 0.72 and f2b < 0.20 and memory["hands"] >= 16

    if obs.to_call == 0:
        if maniac:
            pot_at, half_at = 0.57, 0.51
        elif station:
            pot_at, half_at = 0.60, 0.505
        else:
            pot_at, half_at = 0.66, 0.545
        if ip:
            pot_at -= 0.015
            half_at -= 0.015
        if spr < 2.2:
            pot_at -= 0.03
        if eq >= pot_at and Action.RAISE_POT in clean:
            return _mark_bet(memory, Action.RAISE_POT)
        if eq >= half_at and Action.RAISE_HALF in clean:
            return _mark_bet(memory, Action.RAISE_HALF)
        # Bluff only after this opponent has actually folded to bets.
        if clean and memory["bets"] >= 10 and f2b > 0.40 and obs.raises_this_street == 0:
            freq = min(0.88, (f2b - 0.32) * 1.7)
            if ip:
                freq = min(0.92, freq + 0.08)
            salt = _mix(obs.hand_index, int(obs.street), obs.pot, obs.hole_cards[0], obs.hole_cards[-1])
            if salt < freq:
                action = Action.RAISE_POT if (f2b > 0.62 and Action.RAISE_POT in clean and salt < freq * 0.45) else (
                    Action.RAISE_HALF if Action.RAISE_HALF in clean else Action.RAISE_POT
                )
                return _mark_bet(memory, action)
        return Action.CHECK_CALL

    if eq >= max(0.74, odds + 0.20) and obs.raises_this_street < 3 and Action.RAISE_POT in clean:
        return _mark_bet(memory, Action.RAISE_POT)
    if eq >= max(0.63, odds + 0.12) and obs.raises_this_street < 3 and Action.RAISE_HALF in clean:
        return _mark_bet(memory, Action.RAISE_HALF)
    slack = 0.03 if obs.street != Street.RIVER else 0.0
    if aggro > 0.62:
        slack += 0.04
    if aggro < 0.22 and memory["opp_raises"] + memory["opp_passive"] > 36:
        slack -= 0.045
    if spr < 1.3:
        slack += 0.025
    if eq + slack >= odds:
        return Action.CHECK_CALL
    return _prefer(obs, [Action.FOLD])


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    What you can see in obs:
        hand_index          which hand of the half this is (0, 1, 2, ...)
        street              Street.PREFLOP, Street.FLOP, Street.TURN or Street.RIVER
        hole_cards          your 5 cards as ints, sorted ascending (0-35)
        board               board (community) cards seen so far (0, 3, 4 or 5 ints)
        pot                 total chips both players put in this hand
        to_call             chips you must add to match the opponent
        my_bet, opp_bet     chips you,opponent put in on THIS street
        my_stack, opp_stack chips each player still has behind
        is_button           True if you are the button (small blind)
        legal_actions       the Actions you may return right now
        raise_to            {RAISE_HALF: total, RAISE_POT: total} for legal raises
        min_raise_to        smallest normal raise total (all-ins may be lower)
        raises_this_street  raises so far this street (max 4)
        action_history      this hand's actions so far; each has
                            street, is_me, action, amount, street_total
        big_blind           size of the big blind in chips

    Actions: Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
    Cards are ints: card_rank(c) is 0 (six) .. 8 (ace), card_suit(c) is 0..3,
    and cards_to_str(obs.hole_cards) prints them like "6h 7c 9s Kd Ah".
    Once all 5 board cards are out, omaha_score(obs.hole_cards, obs.board)
    scores your hand (higher is better).
    """
    start = time.perf_counter()
    try:
        return _decide(obs, memory)
    finally:
        memory["spent"] += time.perf_counter() - start


def _pick_tier(obs: Observation, memory: dict) -> int:
    """Open on the normal sample, then reconsider at hands 200, 350, 500, 750.

    The clock is read on every decision. A checkpoint only unlocks the next
    sample size, and only when the time left covers that size for every hand
    still to come. The comparison is a few arithmetic operations, so the extra
    checkpoints do not spend the bank. Falling behind drops back at once.
    """
    hands_left = max(1, memory["num_hands"] - max(0, obs.hand_index))
    allowance = (_TIME_TARGET - memory["spent"]) / hands_left
    if allowance < _NORMAL_AT:
        return 0
    seen = obs.hand_index
    tier = 1
    for hand, step, need in _STEPS:
        if seen >= hand and allowance >= need:
            tier = step
    return tier


def _decide(obs: Observation, memory: dict) -> Action:
    legal = obs.legal_actions
    _note_actions(obs, memory)
    if len(legal) == 1:
        return legal[0]
    memory["tier"] = _pick_tier(obs, memory)
    eq = _raw_equity(obs, memory)
    if obs.street == Street.PREFLOP:
        action = _preflop(obs, eq, memory)
    else:
        action = _postflop(obs, eq, memory)
    if action not in legal:
        return _prefer(obs, [Action.CHECK_CALL, Action.FOLD])
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    memory["hands"] += 1
    if memory["bet_this_hand"]:
        memory["bets"] += 1
        if (not result.went_to_showdown) and result.winner == "me":
            memory["folds_to_bet"] += 1
        memory["bet_this_hand"] = False
    if (not result.went_to_showdown) and result.winner == "me":
        memory["opp_fold_hands"] += 1
    _learn_range(result, memory)


def _learn_range(result: HandResult, memory: dict) -> None:
    """Update P(action | hand class) from a showdown. One hand, one update."""
    if result.hand_index != memory["line_hand"]:
        return
    memory["line_hand"] = -1
    if not result.went_to_showdown or result.opp_hole is None or len(result.board) != 5:
        return
    hole = result.opp_hole
    board = result.board
    kinds: dict[int, int] = {}
    for street, is_me, action, amount in memory["line"]:
        if is_me:
            continue
        if action in (int(Action.RAISE_HALF), int(Action.RAISE_POT)):
            kinds[street] = 2
        elif amount > 0 and kinds.get(street) != 2:
            kinds[street] = 1
        elif street not in kinds:
            kinds[street] = 0
    if memory["line"]:
        street, is_me, action, amount = memory["line"][-1]
        if is_me and action in (int(Action.RAISE_HALF), int(Action.RAISE_POT)) and kinds.get(street) != 2:
            kinds[street] = 1
        elif (
            is_me
            and action == int(Action.CHECK_CALL)
            and amount == 0
            and street == int(Street.RIVER)
            and not memory["saw_button"]
        ):
            kinds.setdefault(street, 0)
    if 0 in kinds:
        memory["pf"][_pf_bucket(hole)][kinds[0]] += 1.0
    if 1 in kinds:
        memory["post"][0][_made_bucket(_partial_score(hole, board[:3]))][kinds[1]] += 1.0
    if 2 in kinds:
        memory["post"][1][_made_bucket(_partial_score(hole, board[:4]))][kinds[2]] += 1.0
    if 3 in kinds:
        memory["post"][2][_made_bucket(_partial_score(hole, board))][kinds[3]] += 1.0

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
