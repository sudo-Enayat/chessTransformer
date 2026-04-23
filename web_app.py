from __future__ import annotations

import time
from pathlib import Path

import chess
from flask import Flask, jsonify, render_template, request

from backends import (
    SeventyEightMBestMoveBackend,
    SeventyEightMMCTSBackend,
    TwelveMBestMoveBackend,
    TwelveMMCTSBackend,
    TwelveMSecondBestBackend,
)


APP_DIR = Path(__file__).resolve().parent

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))


DIFFICULTY_TIERS = [
    {
        "id": "novice",
        "name": "Novice",
        "elo": 800,
        "description": "Plays fast and makes occasional blunders. Perfect for beginners.",
        "backend": "novice",
    },
    {
        "id": "club",
        "name": "Club Player",
        "elo": 1540,
        "description": "Plays solid, fundamental chess. Will punish obvious mistakes.",
        "backend": "club",
    },
    {
        "id": "expert",
        "name": "Expert",
        "elo": 1740,
        "description": "A highly advanced opponent with deep positional understanding.",
        "backend": "expert",
    },
    {
        "id": "grandmaster",
        "name": "Grandmaster",
        "elo": 2500,
        "description": "Searches future timelines to play at a professional Grandmaster level.",
        "backend": "grandmaster",
    },
    {
        "id": "impossible",
        "name": "Impossible",
        "elo": 2900,
        "description": "Superhuman AI, impossible to beat by any human.",
        "backend": "impossible",
    },
]

BACKEND_FACTORIES = {
    "friend_eval": TwelveMBestMoveBackend,
    "analysis_78m": SeventyEightMBestMoveBackend,
    "novice": TwelveMSecondBestBackend,
    "club": TwelveMBestMoveBackend,
    "expert": SeventyEightMBestMoveBackend,
    "grandmaster": TwelveMMCTSBackend,
    "impossible": SeventyEightMMCTSBackend,
}


board_obj = chess.Board()
current_mode = "ai"
current_difficulty = "club"
is_player_white = True
loaded_backends = {}


def get_backend(key: str):
    backend = loaded_backends.get(key)
    if backend is None:
        backend = BACKEND_FACTORIES[key]()
        loaded_backends[key] = backend
    return backend


def get_ai_backend():
    if current_mode != "ai":
        return None
    return get_backend(current_difficulty)


def get_display_eval_backend():
    if current_mode == "friend":
        return get_backend("friend_eval")
    return get_ai_backend() or get_backend("friend_eval")


def get_analysis_backend():
    return get_backend("analysis_78m")


def reset_active_backend() -> None:
    engine = get_ai_backend()
    if engine is not None:
        engine.sync_position(board_obj)


def current_eval() -> float:
    try:
        return get_display_eval_backend().evaluate_position(board_obj)
    except Exception:
        return 0.0


def response_eval(bot_stats: dict | None = None) -> float:
    if bot_stats and "q" in bot_stats:
        try:
            return float(bot_stats["q"])
        except (TypeError, ValueError):
            pass
    return current_eval()


def perform_ai_move() -> tuple[str | None, dict | None]:
    engine = get_ai_backend()
    if engine is None:
        return None, None

    if current_difficulty in {"novice", "club", "expert"}:
        time.sleep(1.0)

    bot_move = engine.choose_move(board_obj)
    if bot_move is None:
        return None, None

    board_obj.push(bot_move)
    engine.advance_root(bot_move)
    return bot_move.uci(), engine.get_last_stats()


def tier_metadata(difficulty_id: str) -> dict:
    for tier in DIFFICULTY_TIERS:
        if tier["id"] == difficulty_id:
            return tier
    raise KeyError(f"Unknown difficulty: {difficulty_id}")


@app.route("/")
def index():
    return render_template("index.html", difficulty_tiers=DIFFICULTY_TIERS)


@app.route("/configure_game", methods=["POST"])
def configure_game():
    global current_mode, current_difficulty

    data = request.get_json(silent=True) or {}
    requested_mode = data.get("mode", "ai")
    requested_difficulty = data.get("difficulty", "club")

    if requested_mode not in {"ai", "friend"}:
        return jsonify({"success": False, "error": "Unsupported mode"}), 400

    try:
        if requested_mode == "friend":
            get_backend("friend_eval").load()
            current_mode = "friend"
            current_difficulty = "club"
        else:
            tier_metadata(requested_difficulty)
            get_backend(requested_difficulty).load()
            current_mode = "ai"
            current_difficulty = requested_difficulty
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify(
        {
            "success": True,
            "mode": current_mode,
            "difficulty": current_difficulty,
            "needs_color_selection": current_mode == "ai",
        }
    )


@app.route("/new_game", methods=["POST"])
def new_game():
    global board_obj, is_player_white

    data = request.get_json(silent=True) or {}
    board_obj = chess.Board()
    bot_move = None
    bot_stats = None

    if current_mode == "ai":
        is_player_white = bool(data.get("play_as_white", True))
        reset_active_backend()
        if not is_player_white:
            bot_move, bot_stats = perform_ai_move()
    else:
        is_player_white = True
        get_backend("friend_eval").load()

    return jsonify(
        {
            "fen": board_obj.fen(),
            "game_over": board_obj.is_game_over(),
            "bot_move": bot_move,
            "bot_stats": bot_stats,
            "eval": response_eval(bot_stats),
            "mode": current_mode,
            "difficulty": current_difficulty,
        }
    )


@app.route("/make_move", methods=["POST"])
def make_move():
    global board_obj

    data = request.get_json(silent=True) or {}
    move_uci = data.get("move")
    if not move_uci:
        return jsonify({"error": "No move provided", "fen": board_obj.fen(), "eval": current_eval()}), 400

    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return jsonify({"error": "Invalid move format", "fen": board_obj.fen(), "eval": current_eval()}), 400

    if move not in board_obj.legal_moves:
        return jsonify({"error": "Illegal move", "fen": board_obj.fen(), "eval": current_eval()}), 400

    board_obj.push(move)
    engine = get_ai_backend()
    if engine is not None:
        engine.advance_root(move)

    if board_obj.is_game_over():
        return jsonify(
            {
                "fen": board_obj.fen(),
                "game_over": True,
                "result": board_obj.result(),
                "bot_move": None,
                "bot_stats": None,
                "eval": current_eval(),
            }
        )

    bot_move = None
    bot_stats = None
    if current_mode == "ai":
        bot_move, bot_stats = perform_ai_move()

    return jsonify(
        {
            "fen": board_obj.fen(),
            "game_over": board_obj.is_game_over(),
            "result": board_obj.result() if board_obj.is_game_over() else None,
            "bot_move": bot_move,
            "bot_stats": bot_stats,
            "eval": response_eval(bot_stats),
        }
    )


@app.route("/bot_status", methods=["GET"])
def bot_status():
    engine = get_ai_backend()
    if engine is None:
        return jsonify(
            {
                "is_thinking": False,
                "sims": 0,
                "depth": 0.0,
                "elapsed": 0.0,
                "limit": 0.0,
                "eval": 0.0,
                "mood": "normal",
            }
        )
    return jsonify(engine.get_status())


@app.route("/undo", methods=["POST"])
def undo():
    global board_obj

    undo_count = 0
    target = 2 if current_mode == "ai" else 1
    while undo_count < target and board_obj.move_stack:
        board_obj.pop()
        undo_count += 1

    reset_active_backend()

    return jsonify({"fen": board_obj.fen(), "eval": current_eval(), "undo_count": undo_count})


@app.route("/resign", methods=["POST"])
def resign():
    global board_obj

    board_obj = chess.Board()
    reset_active_backend()
    return jsonify({"fen": board_obj.fen(), "eval": 0.0})


@app.route("/predict", methods=["GET"])
def predict():
    return jsonify(get_analysis_backend().predict(board_obj))


@app.route("/predict_fen", methods=["POST"])
def predict_fen():
    data = request.get_json(silent=True) or {}
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "No FEN provided"}), 400

    try:
        temp_board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    return jsonify(get_analysis_backend().predict(temp_board))


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
