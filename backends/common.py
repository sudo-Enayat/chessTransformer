from __future__ import annotations


import subprocess
import time
from queue import Empty, Queue
from threading import Thread
from dataclasses import dataclass
from pathlib import Path

import chess
import torch


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent


PIECE_TO_ID = {
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}


@dataclass(frozen=True)
class GreedyConfig:
    model_path: Path
    pick_rank: int = 0
    device: str = "cuda"
    top_moves: int = 3


@dataclass(frozen=True)
class MCTSConfig:
    model_path: Path
    device: str = "cuda"
    normal_time: float = 5.0
    panic_time: float = 15.0
    cpuct: float = 2.5
    cpuct_init: float = 19652.0
    virtual_loss: float = 0.005
    root_noise_alpha: float = 0.0
    root_noise_frac: float = 0.0
    eval_batch_size: int = 128
    replicas: int = 1
    progress_interval: float = 0.5
    cache_capacity: int = 50000
    collect_dup_limit: int = 128
    min_sims: int = 4000
    use_fp32: bool = True


def require_cuda(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "These TorchScript checkpoints were exported for CUDA. "
            "Run this app on a CUDA-enabled machine."
        )
    return torch.device(device_name)


def encode_board(board: chess.Board, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, bool]:
    canonical = board if board.turn == chess.WHITE else board.mirror()
    flipped = board.turn == chess.BLACK
    pcs = torch.zeros((1, 64), dtype=torch.long, device=device)
    for square in chess.SQUARES:
        piece = canonical.piece_at(square)
        if piece is None:
            continue
        pcs[0, square] = PIECE_TO_ID[(piece.color, piece.piece_type)]

    castling = (
        (1 if canonical.has_kingside_castling_rights(chess.WHITE) else 0)
        | (2 if canonical.has_queenside_castling_rights(chess.WHITE) else 0)
        | (4 if canonical.has_kingside_castling_rights(chess.BLACK) else 0)
        | (8 if canonical.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    ep_square = canonical.ep_square if canonical.ep_square is not None else 64
    meta = torch.tensor([[castling, ep_square]], dtype=torch.long, device=device)
    return pcs, meta, flipped


def canonical_token_for_move(move: chess.Move, flipped: bool) -> int:
    from_square = move.from_square ^ 0x38 if flipped else move.from_square
    to_square = move.to_square ^ 0x38 if flipped else move.to_square
    return (from_square * 64) + to_square


def unique_legal_moves(board: chess.Board, flipped: bool) -> tuple[list[chess.Move], list[int]]:
    token_to_move: dict[int, chess.Move] = {}
    for move in board.legal_moves:
        token = canonical_token_for_move(move, flipped)
        current = token_to_move.get(token)
        if current is None:
            token_to_move[token] = move
            continue
        if move.promotion == chess.QUEEN and current.promotion != chess.QUEEN:
            token_to_move[token] = move
    items = list(token_to_move.items())
    return [move for _, move in items], [token for token, _ in items]


def unpack_model_outputs(outputs) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(outputs, (tuple, list)):
        logits = outputs[0]
        values = outputs[1]
    else:
        logits, values = outputs
    return logits, values


def absolute_white_eval(relative_value: float, board: chess.Board) -> float:
    return relative_value if board.turn == chess.WHITE else -relative_value


def build_prediction_payload(
    board: chess.Board,
    logits: torch.Tensor,
    relative_value: float,
    flipped: bool,
    top_n: int,
) -> dict:
    legal_moves, legal_tokens = unique_legal_moves(board, flipped)
    if not legal_moves:
        return {"eval": absolute_white_eval(relative_value, board), "top_moves": []}

    selected_logits = logits[legal_tokens].float()
    probs = torch.softmax(selected_logits, dim=0).detach().cpu().tolist()
    ranked = sorted(zip(legal_moves, probs), key=lambda item: item[1], reverse=True)
    max_prob = ranked[0][1] if ranked else 1.0
    top_moves = []
    for move, prob in ranked[:top_n]:
        top_moves.append(
            {
                "uci": move.uci(),
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "prob": prob,
                "normalized_prob": prob / max_prob if max_prob > 0 else 0.0,
            }
        )

    return {"eval": absolute_white_eval(relative_value, board), "top_moves": top_moves}


class GreedyBackend:
    def __init__(self, config: GreedyConfig):
        self.config = config
        self.device = require_cuda(config.device)
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        self.model = torch.jit.load(str(self.config.model_path), map_location=self.device)
        self.model.eval()

    def sync_position(self, board: chess.Board) -> None:
        _ = board

    def advance_root(self, move: chess.Move) -> None:
        _ = move

    def get_status(self) -> dict:
        return {
            "is_thinking": False,
            "sims": 0,
            "depth": 0.0,
            "elapsed": 0.0,
            "limit": 0.0,
            "eval": 0.0,
            "mood": "normal",
        }

    def get_last_stats(self) -> dict | None:
        return None

    def _evaluate(self, board: chess.Board) -> tuple[torch.Tensor, float, bool]:
        self.load()
        pcs, meta, flipped = encode_board(board, self.device)
        with torch.no_grad():
            logits, values = unpack_model_outputs(self.model(pcs, meta))
        relative_value = float(values.flatten()[0].item())
        return logits[0], relative_value, flipped

    def evaluate_position(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        if board.is_game_over():
            return 0.0

        logits, relative_value, flipped = self._evaluate(board)
        legal_moves, legal_tokens = unique_legal_moves(board, flipped)
        
        if not legal_moves:
            return 0.0
            
        top_k = min(3, len(legal_moves))
        _, top_indices = torch.topk(logits[legal_tokens].float(), top_k)
        
        best_eval = None
        is_white_turn = board.turn == chess.WHITE
        
        for idx in top_indices.tolist():
            move = legal_moves[idx]
            board.push(move)
            
            if board.is_checkmate():
                abs_eval = 1.0 if is_white_turn else -1.0
            elif board.is_game_over():
                abs_eval = 0.0
            else:
                _, next_rel_val, _ = self._evaluate(board)
                abs_eval = absolute_white_eval(next_rel_val, board)
                
            board.pop()
            
            if best_eval is None:
                best_eval = abs_eval
            elif is_white_turn:
                best_eval = max(best_eval, abs_eval)
            else:
                best_eval = min(best_eval, abs_eval)
                
        if best_eval is None:
            return 0.0
        return best_eval

    def predict(self, board: chess.Board) -> dict:
        logits, relative_value, flipped = self._evaluate(board)
        return build_prediction_payload(board, logits, relative_value, flipped, self.config.top_moves)

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        logits, _, flipped = self._evaluate(board)
        legal_moves, legal_tokens = unique_legal_moves(board, flipped)
        if not legal_moves:
            return None

        ranked_indices = sorted(
            range(len(legal_moves)),
            key=lambda idx: float(logits[legal_tokens[idx]].item()),
            reverse=True,
        )
        chosen_rank = min(self.config.pick_rank, len(ranked_indices) - 1)
        return legal_moves[ranked_indices[chosen_rank]]


class NativeMCTSBackend:
    """High-performance MCTS backend using the C++ native_mcts.exe via UCI protocol.

    Instead of running MCTS in Python, this launches the compiled C++ engine as a
    persistent subprocess and communicates via the UCI protocol. The C++ engine
    handles the full search tree, batched GPU model inference, and time management
    all in native code, yielding massive speed-ups over the Python implementation.
    """

    def __init__(self, config: MCTSConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._move_history: list[str] = []
        self._stdout_queue: Queue[str | None] | None = None
        self._stdout_thread: Thread | None = None

        # Live telemetry (updated from search progress lines)
        self._is_thinking = False
        self._sims = 0
        self._depth = 0.0
        self._elapsed = 0.0
        self._limit = 0.0
        self._q = 0.0
        self._last_stats: dict | None = None
        self._last_eval_history: tuple[str, ...] | None = None
        self._last_eval_abs_q = 0.0

        # Resolve exe path (relative to project root)
        self._exe_path = str(APP_DIR / "engines" / "native_mcts.exe")

    # ---- lifecycle ----

    def load(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._start_engine()

    def _start_engine(self) -> None:
        import subprocess as _sp

        cmd = [
            self._exe_path,
            "--model", str(self.config.model_path),
            "--uci",
            "--normal-time", str(self.config.normal_time),
            "--panic-time", str(self.config.panic_time),
            "--cpuct", str(self.config.cpuct),
            "--cpuct-init", str(self.config.cpuct_init),
            "--virtual-loss", str(self.config.virtual_loss),
            "--eval-batch-size", str(self.config.eval_batch_size),
            "--cache-capacity", str(self.config.cache_capacity),
            "--collect-dup-limit", str(self.config.collect_dup_limit),
            "--min-sims", str(self.config.min_sims),
            "--progress-interval", str(self.config.progress_interval),
        ]
        if self.config.use_fp32:
            cmd.append("--fp32")

        # The C++ engine links against LibTorch DLLs. When launched from Flask
        # those DLLs won't be on PATH, causing an immediate silent crash.
        # Locate torch's lib dir via Python's own import and prepend it to PATH.
        import os
        env = os.environ.copy()
        try:
            import torch as _torch
            torch_lib_dir = str(Path(_torch.__file__).parent / "lib")
            env["PATH"] = torch_lib_dir + os.pathsep + env.get("PATH", "")
            print(f"[native] torch lib: {torch_lib_dir}", flush=True)
        except Exception as e:
            print(f"[native] warning: could not locate torch lib: {e}", flush=True)

        print(f"[native] launching: {' '.join(cmd)}", flush=True)

        self._process = _sp.Popen(
            cmd,
            stdin=_sp.PIPE,
            stdout=_sp.PIPE,
            stderr=_sp.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._stdout_queue = Queue()
        self._stdout_thread = Thread(target=self._pump_stdout, daemon=True)
        self._stdout_thread.start()

        # The C++ engine prints uciok on startup before entering the input loop.
        # Just wait for it, then confirm readiness.
        self._read_until("uciok", timeout=30.0)
        self._send("isready")
        self._read_until("readyok", timeout=30.0)
        print("[native] engine ready", flush=True)

    def _pump_stdout(self) -> None:
        assert self._process and self._process.stdout and self._stdout_queue is not None
        try:
            for raw_line in self._process.stdout:
                self._stdout_queue.put(raw_line.rstrip("\r\n"))
        finally:
            self._stdout_queue.put(None)

    def _send(self, line: str) -> None:
        if self._process and self._process.stdin:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()

    def _read_until(self, prefix: str, *, parse_search: bool = False, timeout: float | None = None) -> str:
        """Read stdout lines until one starts with *prefix*.  Returns that line.

        If *parse_search* is True, intermediate ``[search]`` lines are parsed
        to keep live telemetry up-to-date.
        """
        started_at = time.perf_counter()
        while True:
            assert self._process and self._stdout_queue is not None
            remaining = None
            if timeout is not None:
                remaining = max(0.1, timeout - (time.perf_counter() - started_at))
            try:
                line = self._stdout_queue.get(timeout=remaining)
            except Empty:
                raise TimeoutError(f"[native] timed out waiting for {prefix!r}")
            if line is None:
                return_code = self._process.poll()
                if return_code is None:
                    raise RuntimeError("[native] engine output stream closed unexpectedly")
                raise RuntimeError(f"[native] engine process terminated unexpectedly (exit code {return_code})")
            if parse_search and "[search]" in line:
                self._parse_search_line(line)
            if line.startswith(prefix):
                return line

    def _parse_search_line(self, line: str) -> None:
        """Parse ``[search] 2.45/5.00s sims=1234 …`` telemetry."""
        try:
            parts = line.split()
            for part in parts:
                if "sims=" in part:
                    self._sims = int(part.split("=")[1])
                elif part.startswith("q="):
                    self._q = float(part.split("=")[1])
                elif "depth=" in part and "=" in part:
                    depth_str = part.split("=")[1].split("/")
                    self._depth = float(depth_str[0])
                elif "/" in part and part.endswith("s"):
                    time_parts = part.rstrip("s").split("/")
                    if len(time_parts) == 2:
                        self._elapsed = float(time_parts[0])
                        self._limit = float(time_parts[1])
        except Exception:
            pass

    # ---- public interface (matches old MCTSBackend) ----

    def sync_position(self, board: chess.Board) -> None:
        self.load()
        self._move_history = [m.uci() for m in board.move_stack]
        self._last_eval_history = None
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def advance_root(self, move: chess.Move) -> None:
        self._move_history.append(move.uci())

    def get_status(self) -> dict:
        is_white_turn = (len(self._move_history) % 2 == 0)
        abs_q = self._q if is_white_turn else -self._q
        return {
            "is_thinking": self._is_thinking,
            "sims": self._sims,
            "depth": self._depth,
            "elapsed": round(self._elapsed, 2),
            "limit": round(self._limit, 2),
            "eval": abs_q,
            "mood": "normal",
        }

    def get_last_stats(self) -> dict | None:
        return self._last_stats

    def evaluate_position(self, board: chess.Board) -> float:
        """Return the last known Q value if the board matches our history, or 0.0."""
        self.load()
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        if board.is_game_over():
            return 0.0

        expected_moves = tuple(m.uci() for m in board.move_stack)
        if expected_moves == self._last_eval_history and self._last_stats is not None:
            return self._last_eval_abs_q

        return 0.0

    def predict(self, board: chess.Board) -> dict:
        """Predict with a greedy model fallback — MCTS engines don't expose raw logits."""
        return {"eval": 0.0, "top_moves": []}

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        self.load()

        # If the board doesn't match our history, re-sync
        expected_moves = [m.uci() for m in board.move_stack]
        if expected_moves != self._move_history:
            self._move_history = expected_moves

        # Send position
        if self._move_history:
            pos_cmd = "position startpos moves " + " ".join(self._move_history)
        else:
            pos_cmd = "position startpos"
        self._send(pos_cmd)

        # Search
        self._is_thinking = True
        self._sims = 0
        self._depth = 0.0
        self._elapsed = 0.0
        self._limit = self.config.normal_time
        self._q = 0.0
        self._last_eval_history = None

        try:
            self._send("go")
            bestmove_line = self._read_until(
                "bestmove",
                parse_search=True,
                timeout=max(self.config.panic_time, self.config.normal_time) + 60.0,
            )
        except Exception:
            if self._process and self._process.poll() is None:
                try:
                    self._send("quit")
                    self._process.wait(timeout=2)
                except Exception:
                    self._process.kill()
            self._process = None
            self._stdout_queue = None
            self._stdout_thread = None
            self._last_stats = None
            self._last_eval_history = None
            raise
        finally:
            self._is_thinking = False

        # Parse bestmove
        parts = bestmove_line.split()
        if len(parts) < 2 or parts[1] == "(none)":
            self._last_stats = None
            return None

        move_uci = parts[1]
        is_white_turn = (len(self._move_history) % 2 == 0)
        abs_q = self._q if is_white_turn else -self._q
        self._last_eval_history = tuple(self._move_history)
        self._last_eval_abs_q = abs_q
        self._last_stats = {
            "sims": self._sims,
            "depth": round(self._depth, 1),
            "time": round(self._elapsed, 1),
            "q": round(abs_q, 2),
            "mood": "normal",
        }

        try:
            return chess.Move.from_uci(move_uci)
        except ValueError:
            return None

    def __del__(self):
        if self._process and self._process.poll() is None:
            try:
                self._send("quit")
                self._process.wait(timeout=3)
            except Exception:
                self._process.kill()


# Alias — the web-app tier classes use MCTSBackend, point them at the native one.
MCTSBackend = NativeMCTSBackend
