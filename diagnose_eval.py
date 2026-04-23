# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Diagnostic script: probe both 12M and 78M models on critical positions
to understand the eval bar's behaviour near mate / winning / losing.

Run from the ChessTransformer directory:
    python diagnose_eval.py
"""

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
from pathlib import Path

import chess
import torch

# ── make sure imports resolve ──
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from backends.common import (
    APP_DIR as _APP,
    encode_board,
    unpack_model_outputs,
    unique_legal_moves,
    absolute_white_eval,
    canonical_token_for_move,
)

# ─── Model paths ───
MODEL_12M = _APP / "models" / "SSChess_12M.pt"
MODEL_78M = _APP / "models" / "SSChess_78M_FP32.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Test positions ───
# Each entry: (label, FEN, description)
TEST_POSITIONS = [
    # ── Starting position (should be ~0.0) ──
    ("Starting Position", chess.STARTING_FEN, "Equal, game just began"),

    # ── White has mate in 1 (Qh7#) ──
    ("White Mate-in-1",
     "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
     "Scholar's mate setup - White plays Qxf7#"),

    # ── Black has mate in 1 ──
    ("Black Mate-in-1",
     "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR b KQkq - 1 3",
     "Black plays Qxe1# or similar - Black is winning decisively"),

    # ── White is up a queen (massive material advantage) ──
    ("White +Queen",
     "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "White has all pieces, Black lost the queen"),

    # ── Dead drawn endgame (K vs K) ──
    ("K vs K Draw",
     "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
     "King vs King - dead draw"),

    # ── White is getting checkmated (in check, only move leads to mate) ──
    ("White about to be mated",
     "r1bqkbnr/pppppppp/2n5/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
     "White down material, weak position"),

    # ── Obvious winning endgame for White (K+Q vs K) ──
    ("White K+Q vs K",
     "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
     "White has K+Q vs lone K - should be near +1.0"),

    # ── Tricky: White has mate in 1 but from black's perspective the position looks ok ──
    ("Fool's Mate Setup",
     "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1",
     "After 1.f3 - Black can play e5, heading toward fool's mate"),

    # ── Simulate: position AFTER the mating move (game over) ──
    ("Post-Checkmate Position",
     "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
     "White just played Qxf7# - Black is in checkmate"),
]


def load_model(path: Path):
    print(f"  Loading {path.name} ... ", end="", flush=True)
    model = torch.jit.load(str(path), map_location=DEVICE)
    model.eval()
    print("OK")
    return model


def probe_model(model, board: chess.Board, model_name: str) -> dict:
    """Run a single forward pass and return raw diagnostics."""
    pcs, meta, flipped = encode_board(board, DEVICE)

    with torch.no_grad():
        logits, values = unpack_model_outputs(model(pcs, meta))

    raw_value = float(values.flatten()[0].item())
    abs_eval = absolute_white_eval(raw_value, board)

    # Get top-3 moves by policy
    legal_moves, legal_tokens = unique_legal_moves(board, flipped)
    top_moves = []
    if legal_moves:
        selected_logits = logits[0][legal_tokens].float()
        probs = torch.softmax(selected_logits, dim=0).detach().cpu().tolist()
        ranked = sorted(zip(legal_moves, probs), key=lambda x: x[1], reverse=True)
        for move, prob in ranked[:5]:
            top_moves.append((move.uci(), prob))

    return {
        "model": model_name,
        "raw_value": raw_value,
        "abs_white_eval": abs_eval,
        "turn": "White" if board.turn == chess.WHITE else "Black",
        "top_moves": top_moves,
    }


def probe_lookahead(model, board: chess.Board, model_name: str) -> dict | None:
    """
    For each legal move, make the move, evaluate the resulting position,
    and report the range of evals seen. This simulates a 1-ply lookahead.
    """
    legal_moves, _ = unique_legal_moves(board, False)  # flipped doesn't matter here
    if not legal_moves:
        return None

    evals = []
    for move in board.legal_moves:
        board.push(move)
        pcs, meta, flipped = encode_board(board, DEVICE)
        with torch.no_grad():
            _, values = unpack_model_outputs(model(pcs, meta))
        raw_v = float(values.flatten()[0].item())
        # After pushing, board.turn flipped, so absolute_white_eval uses the NEW turn
        abs_v = absolute_white_eval(raw_v, board)
        evals.append((move.uci(), abs_v))
        board.pop()

    evals.sort(key=lambda x: x[1], reverse=True)  # best for White first

    return {
        "model": model_name,
        "best_move_eval": evals[0],
        "worst_move_eval": evals[-1],
        "eval_range": evals[0][1] - evals[-1][1],
        "top_3": evals[:3],
        "bottom_3": evals[-3:],
        "total_moves": len(evals),
    }


def main():
    print("=" * 70)
    print("  EVAL DIAGNOSTIC: Probing 12M and 78M value heads")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print()

    model_12m = load_model(MODEL_12M)
    model_78m = load_model(MODEL_78M)
    print()

    for label, fen, description in TEST_POSITIONS:
        board = chess.Board(fen)
        print("-" * 70)
        print(f"  >> {label}")
        print(f"     {description}")
        print(f"     FEN: {fen}")
        print(f"     Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        print(f"     Game Over: {board.is_game_over()}  |  In Check: {board.is_check()}")
        if board.is_checkmate():
            print(f"     *** CHECKMATE ***")
        print()

        for model, name in [(model_12m, "12M"), (model_78m, "78M")]:
            result = probe_model(model, board, name)
            print(f"     [{name}] Raw Value Head: {result['raw_value']:+.6f}")
            print(f"     [{name}] Absolute White Eval: {result['abs_white_eval']:+.6f}")
            if result["top_moves"]:
                moves_str = "  |  ".join(
                    f"{uci} ({prob:.1%})" for uci, prob in result["top_moves"][:3]
                )
                print(f"     [{name}] Top moves: {moves_str}")
            print()

        # 1-ply lookahead comparison (only if game not over)
        if not board.is_game_over():
            print(f"     -- 1-ply Lookahead --")
            for model, name in [(model_12m, "12M"), (model_78m, "78M")]:
                la = probe_lookahead(model, board, name)
                if la:
                    best_uci, best_v = la["best_move_eval"]
                    worst_uci, worst_v = la["worst_move_eval"]
                    print(f"     [{name}] Best  after-move: {best_uci} -> eval {best_v:+.6f}")
                    print(f"     [{name}] Worst after-move: {worst_uci} -> eval {worst_v:+.6f}")
                    print(f"     [{name}] Eval spread: {la['eval_range']:.6f}  ({la['total_moves']} legal moves)")
            print()

    # ── Summary table ──
    print("=" * 70)
    print("  SUMMARY TABLE: Raw value-head outputs")
    print("=" * 70)
    header = f"  {'Position':<30} {'Turn':<6} {'12M raw':>10} {'12M abs':>10} {'78M raw':>10} {'78M abs':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, fen, _ in TEST_POSITIONS:
        board = chess.Board(fen)
        r12 = probe_model(model_12m, board, "12M")
        r78 = probe_model(model_78m, board, "78M")
        turn = "W" if board.turn == chess.WHITE else "B"
        print(
            f"  {label:<30} {turn:<6} "
            f"{r12['raw_value']:>+10.4f} {r12['abs_white_eval']:>+10.4f} "
            f"{r78['raw_value']:>+10.4f} {r78['abs_white_eval']:>+10.4f}"
        )
    print("=" * 70)

    print()
    print("  KEY OBSERVATIONS TO LOOK FOR:")
    print("  1. Does the value head output approach +/-1.0 for mate positions?")
    print("  2. Is there a big difference between 12M and 78M?")
    print("  3. Does 1-ply lookahead dramatically change the eval?")
    print("     (If yes -> the static eval is weak, but lookahead helps)")
    print("  4. Is the eval 'flat' even in decisive positions?")
    print("     (If yes → the value head was under-trained or lb_coeff too low)")
    print()


if __name__ == "__main__":
    main()
