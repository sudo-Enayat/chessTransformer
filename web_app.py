from __future__ import annotations

import time
from pathlib import Path
from threading import Lock, Thread, Timer

import chess
from flask import Flask, jsonify, render_template, request

from backends import (
    SeventyEightMBestMoveBackend,
    SeventyEightMMCTSBackend,
    SeventyEightMMCTSRefereeBackend,
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
        "elo": 1760,
        "description": "A highly advanced opponent with deep positional understanding.",
        "backend": "expert",
    },
    {
        "id": "grandmaster",
        "name": "Grandmaster",
        "elo": 2500,
        "description": "Searches future timelines to play at a Grandmaster level.",
        "backend": "grandmaster",
    },
    {
        "id": "impossible",
        "name": "Impossible",
        "elo": 3200,
        "description": "Superhuman AI, impossible to beat by any human.",
        "backend": "impossible",
    },
]

BACKEND_FACTORIES = {
    "friend_eval": SeventyEightMMCTSRefereeBackend,
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
game_active = False
loaded_backends = {}
backend_load_jobs = {}
backend_load_lock = Lock()
game_lock = Lock()
friend_evals = [0.0]


def stop_thinking_if_any():
    engine = get_ai_backend()
    if engine is not None:
        try:
            engine.stop_search()
        except Exception:
            pass


def get_backend(key: str):
    backend = loaded_backends.get(key)
    if backend is None:
        backend = BACKEND_FACTORIES[key]()
        loaded_backends[key] = backend
    return backend


def unload_backend(key: str) -> None:
    backend = loaded_backends.pop(key, None)
    with backend_load_lock:
        backend_load_jobs.pop(key, None)
    if backend is not None and hasattr(backend, "unload"):
        Thread(target=backend.unload, daemon=True).start()


def unload_inactive_game_backends(next_key: str | None) -> None:
    for key in list(loaded_backends):
        if key in {"friend_eval", "analysis_78m"}:
            continue
        if key != next_key:
            unload_backend(key)


def get_ai_backend():
    if current_mode != "ai":
        return None
    return get_backend(current_difficulty)


def backend_is_loaded(key: str) -> bool:
    backend = loaded_backends.get(key)
    if backend is None:
        return False
    if hasattr(backend, "is_loaded"):
        return bool(backend.is_loaded())
    return True


def start_backend_load(key: str) -> None:
    backend = get_backend(key)
    if backend_is_loaded(key):
        return

    with backend_load_lock:
        job = backend_load_jobs.get(key)
        if job and job.get("status") == "loading":
            return
        backend_load_jobs[key] = {"status": "loading", "error": None}

    def load_worker() -> None:
        try:
            backend.load()
        except Exception as exc:
            with backend_load_lock:
                backend_load_jobs[key] = {"status": "error", "error": str(exc)}
            return
        with backend_load_lock:
            backend_load_jobs[key] = {"status": "ready", "error": None}

    # Let the configure request return and the browser paint the next setup
    # screen before Torch/LibTorch startup begins competing for resources.
    timer = Timer(0.25, load_worker)
    timer.daemon = True
    timer.start()


def backend_load_state(key: str) -> dict:
    if backend_is_loaded(key):
        return {"status": "ready", "error": None}
    with backend_load_lock:
        job = backend_load_jobs.get(key)
        if job:
            return dict(job)
    return {"status": "idle", "error": None}


def ensure_backend_ready(key: str) -> tuple[bool, dict]:
    state = backend_load_state(key)
    if state["status"] == "ready":
        return True, state
    if state["status"] in {"idle", "error"}:
        start_backend_load(key)
        state = backend_load_state(key)
    return state["status"] == "ready", state


def get_display_eval_backend():
    if current_mode == "friend":
        return get_backend("friend_eval")
    return get_ai_backend() or get_backend("friend_eval")


def get_analysis_backend():
    return get_backend("analysis_78m")


def selected_backend_key() -> str:
    return "friend_eval" if current_mode == "friend" else current_difficulty


def reset_active_backend() -> None:
    engine = get_ai_backend()
    if engine is not None:
        engine.sync_position(board_obj)


def current_eval() -> float:
    if current_mode == "friend":
        return friend_evals[-1] if friend_evals else 0.0
    try:
        key = selected_backend_key()
        if not backend_is_loaded(key):
            return 0.0
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


def terminal_white_eval(board: chess.Board) -> float | None:
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_game_over():
        return 0.0
    return None


def immediate_checkmate_move(board: chess.Board) -> chess.Move | None:
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move
    return None


def first_bot_move_should_be_greedy() -> bool:
    return current_mode == "ai" and current_difficulty == "impossible" and len(board_obj.move_stack) <= 1


def simple_bot_stats(move_uci: str, q: float, mood: str = "normal") -> dict:
    return {
        "sims": 0,
        "depth": 0.0,
        "max_depth": 0.0,
        "time": 0.0,
        "q": round(q, 2),
        "mood": mood,
        "move": move_uci,
    }


def perform_ai_move() -> tuple[str | None, dict | None]:
    engine = get_ai_backend()
    if engine is None:
        return None, None

    if current_difficulty in {"novice", "club", "expert"}:
        time.sleep(1.0)

    stats_override = None
    bot_move = immediate_checkmate_move(board_obj)
    if bot_move is not None:
        stats_override = simple_bot_stats(bot_move.uci(), 1.0 if board_obj.turn == chess.WHITE else -1.0, "mate")
    elif first_bot_move_should_be_greedy():
        greedy = get_analysis_backend()
        bot_move = greedy.choose_move(board_obj)
        if bot_move is not None:
            stats_override = simple_bot_stats(bot_move.uci(), 0.0, "greedy")
    else:
        bot_move = engine.choose_move(board_obj)

    if bot_move is not None and bot_move not in board_obj.legal_moves:
        engine.sync_position(board_obj)
        bot_move = engine.choose_move(board_obj)
        stats_override = None
    if bot_move is not None and bot_move not in board_obj.legal_moves:
        raise RuntimeError(f"Engine returned illegal move: {bot_move.uci()} for {board_obj.fen()}")
    if bot_move is None:
        return None, None

    board_obj.push(bot_move)
    engine.advance_root(bot_move)
    bot_stats = stats_override or engine.get_last_stats()
    exact_eval = terminal_white_eval(board_obj)
    if exact_eval is not None:
        if bot_stats is None:
            bot_stats = simple_bot_stats(bot_move.uci(), exact_eval, "terminal")
        else:
            bot_stats = dict(bot_stats)
            bot_stats["q"] = round(exact_eval, 2)
            bot_stats["mood"] = "mate" if board_obj.is_checkmate() else bot_stats.get("mood", "terminal")
    return bot_move.uci(), bot_stats


def tier_metadata(difficulty_id: str) -> dict:
    for tier in DIFFICULTY_TIERS:
        if tier["id"] == difficulty_id:
            return tier
    raise KeyError(f"Unknown difficulty: {difficulty_id}")


def board_history_uci() -> list[str]:
    return [move.uci() for move in board_obj.move_stack]


def game_state_payload(extra: dict | None = None) -> dict:
    game_over = board_obj.is_game_over()
    payload = {
        "active": game_active,
        "fen": board_obj.fen(),
        "moves": board_history_uci(),
        "game_over": game_over,
        "result": board_obj.result() if game_over else None,
        "eval": current_eval() if game_active else 0.0,
        "mode": current_mode,
        "difficulty": current_difficulty,
        "is_player_white": is_player_white,
    }
    if extra:
        payload.update(extra)
    return payload


@app.route("/")
def index():
    return render_template("index.html", difficulty_tiers=DIFFICULTY_TIERS)


@app.route("/state", methods=["GET"])
def state():
    with game_lock:
        return jsonify(game_state_payload())


@app.route("/configure_game", methods=["POST"])
def configure_game():
    global board_obj, current_mode, current_difficulty, game_active, friend_evals

    stop_thinking_if_any()
    with game_lock:
        data = request.get_json(silent=True) or {}
        requested_mode = data.get("mode", "ai")
        requested_difficulty = data.get("difficulty", "club")

        if requested_mode not in {"ai", "friend"}:
            return jsonify({"success": False, "error": "Unsupported mode"}), 400

        try:
            if requested_mode == "friend":
                if current_mode != "friend":
                    unload_inactive_game_backends(None)
                start_backend_load("friend_eval")
                current_mode = "friend"
                current_difficulty = "club"
            else:
                tier_metadata(requested_difficulty)
                if current_mode != "ai" or current_difficulty != requested_difficulty:
                    unload_inactive_game_backends(requested_difficulty)
                start_backend_load(requested_difficulty)
                current_mode = "ai"
                current_difficulty = requested_difficulty
            board_obj = chess.Board()
            game_active = False
            friend_evals = [0.0]
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

        return jsonify(
            {
                "success": True,
                "mode": current_mode,
                "difficulty": current_difficulty,
                "needs_color_selection": current_mode == "ai",
                "load_status": backend_load_state(selected_backend_key()),
            }
        )


@app.route("/load_status", methods=["GET"])
def load_status():
    return jsonify(backend_load_state(selected_backend_key()))


@app.route("/new_game", methods=["POST"])
def new_game():
    global board_obj, is_player_white, game_active, friend_evals

    stop_thinking_if_any()
    with game_lock:
        data = request.get_json(silent=True) or {}
        backend_key = selected_backend_key()
        ready, load_state = ensure_backend_ready(backend_key)
        if not ready:
            return jsonify(
                {
                    "loading": True,
                    "load_status": load_state,
                    "mode": current_mode,
                    "difficulty": current_difficulty,
                }
            ), 202

        board_obj = chess.Board()
        game_active = True
        friend_evals = [0.0]
        bot_move = None
        bot_stats = None

        if current_mode == "ai":
            is_player_white = bool(data.get("play_as_white", True))
            reset_active_backend()
            if not is_player_white:
                try:
                    bot_move, bot_stats = perform_ai_move()
                except Exception as exc:
                    game_active = False
                    return jsonify({"error": str(exc), "fen": board_obj.fen(), "eval": current_eval()}), 500
        else:
            is_player_white = True
            get_backend("friend_eval").load()

        return jsonify(
            {
                "fen": board_obj.fen(),
                "game_over": board_obj.is_game_over(),
                "result": board_obj.result() if board_obj.is_game_over() else None,
                "bot_move": bot_move,
                "bot_stats": bot_stats,
                "eval": response_eval(bot_stats),
                "mode": current_mode,
                "difficulty": current_difficulty,
            }
        )


@app.route("/make_move", methods=["POST"])
def make_move():
    global board_obj, game_active, friend_evals

    data = request.get_json(silent=True) or {}
    move_uci = data.get("move")
    if not move_uci:
        return jsonify({"error": "No move provided", "fen": board_obj.fen(), "eval": current_eval()}), 400

    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return jsonify({"error": "Invalid move format", "fen": board_obj.fen(), "eval": current_eval()}), 400

    with game_lock:
        if move not in board_obj.legal_moves:
            return jsonify({"error": "Illegal move", "fen": board_obj.fen(), "eval": current_eval()}), 400

        prev_eval = friend_evals[-1] if friend_evals else 0.0

        board_obj.push(move)
        game_active = True
        engine = get_ai_backend()
        if engine is not None:
            engine.advance_root(move)

        # In vs friend mode, compute evaluation after the move is pushed
        friend_stats = None
        if current_mode == "friend":
            curr_eval = 0.0
            if board_obj.is_checkmate():
                curr_eval = 1.0 if board_obj.turn == chess.BLACK else -1.0
            elif board_obj.is_game_over():
                curr_eval = 0.0
            else:
                referee = get_backend("friend_eval")
                referee.choose_move(board_obj)
                stats = referee.get_last_stats()
                if stats and "q" in stats:
                    curr_eval = stats["q"]
                else:
                    curr_eval = 0.0

            friend_evals.append(curr_eval)

            if board_obj.turn == chess.BLACK:
                delta = curr_eval - prev_eval
            else:
                delta = prev_eval - curr_eval

            is_blunder = delta <= -0.15
            friend_stats = {
                "q": round(curr_eval, 3),
                "delta": round(delta, 3),
                "is_blunder": is_blunder,
                "completed_full_move": (len(board_obj.move_stack) % 2 == 0),
            }

        if board_obj.is_game_over():
            return jsonify(
                {
                    "fen": board_obj.fen(),
                    "game_over": True,
                    "result": board_obj.result(),
                    "bot_move": None,
                    "bot_stats": None,
                    "eval": current_eval(),
                    "friend_stats": friend_stats,
                }
            )

        bot_move = None
        bot_stats = None
        if current_mode == "ai":
            try:
                bot_move, bot_stats = perform_ai_move()
            except Exception as exc:
                return jsonify({"error": str(exc), "fen": board_obj.fen(), "eval": current_eval()}), 500

        return jsonify(
            {
                "fen": board_obj.fen(),
                "game_over": board_obj.is_game_over(),
                "result": board_obj.result() if board_obj.is_game_over() else None,
                "bot_move": bot_move,
                "bot_stats": bot_stats,
                "eval": response_eval(bot_stats),
                "friend_stats": friend_stats,
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
    global board_obj, is_player_white, friend_evals

    stop_thinking_if_any()
    with game_lock:
        undo_count = 0
        target = 2 if current_mode == "ai" else 1
        while undo_count < target and board_obj.move_stack:
            board_obj.pop()
            undo_count += 1

        if current_mode == "friend":
            for _ in range(undo_count):
                if len(friend_evals) > 1:
                    friend_evals.pop()

        reset_active_backend()
        bot_move = None
        bot_stats = None
        if current_mode == "ai" and board_obj.turn != is_player_white and not board_obj.is_game_over():
            bot_move, bot_stats = perform_ai_move()

        return jsonify(
            {
                "fen": board_obj.fen(),
                "eval": response_eval(bot_stats),
                "undo_count": undo_count,
                "bot_move": bot_move,
                "bot_stats": bot_stats,
                "game_over": board_obj.is_game_over(),
                "result": board_obj.result() if board_obj.is_game_over() else None,
            }
        )


@app.route("/resign", methods=["POST"])
def resign():
    global board_obj, game_active

    stop_thinking_if_any()
    with game_lock:
        board_obj = chess.Board()
        game_active = False
        reset_active_backend()
        return jsonify(game_state_payload({"eval": 0.0}))


@app.route("/predict", methods=["GET"])
def predict():
    with game_lock:
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

    with game_lock:
        return jsonify(get_analysis_backend().predict(temp_board))


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
