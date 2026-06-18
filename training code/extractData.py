# extracts data from a folder of pgn files

import glob
import math
import multiprocessing as mp
import os
import re
import shutil
import time

import chess
import chess.pgn
import numpy as np

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = SCRIPT_DIR
folders = ["3200plus","3300plus","3400plus", "3500plus", "3600plus"]

VAL_GAMES_COUNT = 8000
WORKERS = 6
POSITIONS_PER_FILE = 750000

# Defensive casting: prevents string values from notebook cells / config edits
# from reaching numeric comparisons or worker chunking.
VAL_GAMES_COUNT = int(VAL_GAMES_COUNT)
WORKERS = int(WORKERS)
POSITIONS_PER_FILE = int(POSITIONS_PER_FILE)

# Directories
train_dir = os.path.join(base_dir, "train_data")
val_dir = os.path.join(base_dir, "val_data")
golden_dir = os.path.join(base_dir, "golden_data")

# Neural Net Dtype
dtype = [("pcs", np.uint8, (64,)), ("meta", np.uint8, (2,)), ("y", np.uint16), ("v", np.int16)]

# Regex
BRACKET_EVAL_REGEX = re.compile(r"\[%eval\s+([^\]]+)\]", re.IGNORECASE)
PLAIN_EVAL_REGEX = re.compile(
    r"(?:^|[\s\)])([+\-]?(?:[Mm]\d+|#\d+|\d+(?:\.\d+)?)|[Mm][+\-]?\d+)\s*/\s*\d+",
    re.IGNORECASE,
)
HASH_MATE_REGEX = re.compile(r"([+\-]?#[0-9]+)", re.IGNORECASE)
OPENING_REGEX = re.compile(r"([A-E]\d{2})[:\s]")
DRAW_REGEX = re.compile(r"\bDRAW\b", re.IGNORECASE)
WHITE_WIN_REGEX = re.compile(r"\bWHITE WINS\b", re.IGNORECASE)
BLACK_WIN_REGEX = re.compile(r"\bBLACK WINS\b", re.IGNORECASE)


def result_to_cp(result):
    if result == "1-0":
        return 15000
    if result == "0-1":
        return -15000
    if result == "1/2-1/2":
        return 0
    return None


def parse_eval_token(token, is_percent_eval=False):
    token = token.strip()
    if not token:
        return None, False

    token_upper = token.upper()
    if token_upper.startswith("M+") or token_upper.startswith("M-"):
        return (15000 if token_upper[1] == "+" else -15000), True
    if token_upper.startswith(("+M", "-M", "M")) or "#" in token_upper:
        return (15000 if not token.startswith("-") else -15000), True

    try:
        numeric = float(token)
    except ValueError:
        return None, False

    # CCRL inline evals are pawns, while [%eval 76, 27] is already centipawns.
    cp = int(round(numeric if (is_percent_eval and "." not in token) else numeric * 100))
    cp = max(min(cp, 15000), -15000)
    return cp, abs(cp) >= 15000


def extract_comment_value(comment, board_turn, game_result=None):
    if not comment:
        return None, False, False

    comment = comment.strip()
    if not comment:
        return None, False, False

    comment_upper = comment.upper()
    if "BOOK" in comment_upper or OPENING_REGEX.search(comment_upper):
        return 0, True, False

    bracket_match = BRACKET_EVAL_REGEX.search(comment)
    if bracket_match:
        eval_token = bracket_match.group(1).split(",", 1)[0]
        val, is_mate = parse_eval_token(eval_token, is_percent_eval=True)
        if val is not None:
            return val, False, is_mate

    plain_match = PLAIN_EVAL_REGEX.search(comment)
    if plain_match:
        val, is_mate = parse_eval_token(plain_match.group(1), is_percent_eval=False)
        if val is not None:
            return val, False, is_mate

    mate_match = HASH_MATE_REGEX.search(comment)
    if mate_match:
        val, is_mate = parse_eval_token(mate_match.group(1), is_percent_eval=True)
        if val is not None:
            return val, False, is_mate

    result_cp = result_to_cp(game_result)
    if result_cp is not None:
        is_term = False
        if DRAW_REGEX.search(comment_upper):
            is_term = True
        elif WHITE_WIN_REGEX.search(comment_upper) or BLACK_WIN_REGEX.search(comment_upper):
            is_term = True
        elif "ADJUDICATION" in comment_upper:
            is_term = True
            
        if is_term:
            if board_turn == chess.BLACK:
                result_cp = -result_cp
            return result_cp, False, abs(result_cp) >= 15000

    return None, False, False


def get_canonical_data(board, move, comment, game_result):
    """Relative WDL Eval and canonical board state for side-to-move training."""
    val, is_book, is_mate = extract_comment_value(comment, board.turn, game_result=game_result)
    if val is None:
        return None, False

    wdl_val = math.tanh(val / 400.0)
    v_target = int(wdl_val * 10000)

    pcs = np.zeros(64, dtype=np.uint8)
    flip = board.turn == chess.BLACK

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        piece_color = (not piece.color) if flip else piece.color
        piece_type = piece.piece_type + (6 if piece_color == chess.BLACK else 0)
        mapped_sq = sq ^ 0x38 if flip else sq
        pcs[mapped_sq] = piece_type

    if not flip:
        cast = (
            (1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
            | (2 if board.has_queenside_castling_rights(chess.WHITE) else 0)
            | (4 if board.has_kingside_castling_rights(chess.BLACK) else 0)
            | (8 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        )
        ep = board.ep_square if board.ep_square is not None else 64
    else:
        cast = (
            (1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
            | (2 if board.has_queenside_castling_rights(chess.BLACK) else 0)
            | (4 if board.has_kingside_castling_rights(chess.WHITE) else 0)
            | (8 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        )
        ep = (board.ep_square ^ 0x38) if board.ep_square is not None else 64

    from_sq, to_sq = move.from_square, move.to_square
    if flip:
        from_sq ^= 0x38
        to_sq ^= 0x38
    y_target = np.uint16((from_sq * 64) + to_sq)

    return (pcs, cast, ep, y_target, v_target, is_mate), is_book


def is_golden_position(is_elite, is_book, is_mate):
    """
    Golden data is reserved for:
    - 3500plus and 3600plus games
    - book/opening moves
    - mate scores only
    """
    return is_elite or is_book or is_mate


def fast_scan_pgn_offsets(filepath):
    """Scans whole file for game starts. No limits."""
    offsets = []
    with open(filepath, "rb") as handle:
        offset = 0
        for line in handle:
            if line.startswith(b"[Event "):
                offsets.append(offset)
            offset += len(line)
    return offsets


def split_tasks(tasks, num_chunks):
    if not tasks:
        return []
    if num_chunks <= 1:
        return [tasks]

    chunk_size = math.ceil(len(tasks) / num_chunks)
    return [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]


def merge_fragments(directory, pattern, positions_per_file, mem_dtype, prefix):
    """Combines scattered `_end.bin` files into uniform chunks of maximum size."""
    fragment_files = glob.glob(os.path.join(directory, pattern))
    if not fragment_files:
        return

    # Load all fragments and clear up disk space simultaneously
    arrays = []
    for f in fragment_files:
        arrays.append(np.fromfile(f, dtype=mem_dtype))
        os.remove(f)

    if not arrays:
        return

    combined = np.concatenate(arrays)
    total_positions = len(combined)

    num_full = total_positions // positions_per_file

    # Write out full max-sized bin files
    for i in range(num_full):
        chunk = combined[i * positions_per_file : (i + 1) * positions_per_file]
        chunk.tofile(os.path.join(directory, f"{prefix}_merged_{i}.bin"))

    # Write out any remaining bits as exactly one end file
    remainder = total_positions % positions_per_file
    if remainder > 0:
        chunk = combined[num_full * positions_per_file :]
        chunk.tofile(os.path.join(directory, f"{prefix}_merged_end.bin"))


class NullQueue:
    def put(self, _msg):
        pass


def worker_process(worker_id, tasks, queue):
    train_buf = np.zeros(POSITIONS_PER_FILE, dtype=dtype)
    gold_buf = np.zeros(POSITIONS_PER_FILE, dtype=dtype)

    counts = {"train": 0, "gold": 0}
    f_idx = {"train": 0, "gold": 0}

    move_total, game_total = 0, 0

    for filepath, offset, is_elite in tasks:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as pgn:
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            if not game:
                continue

            board = game.board()
            game_result = game.headers.get("Result", "*")
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                data, is_book = get_canonical_data(board, next_node.move, next_node.comment, game_result)

                if data:
                    pcs, cast, ep, target, v, is_mate = data

                    train_buf[counts["train"]] = (pcs, [cast, ep], target, v)
                    counts["train"] += 1
                    if counts["train"] >= POSITIONS_PER_FILE:
                        train_buf.tofile(os.path.join(train_dir, f"worker{worker_id}_train{f_idx['train']}.bin"))
                        f_idx["train"] += 1
                        counts["train"] = 0

                    if is_golden_position(is_elite, is_book, is_mate):
                        gold_buf[counts["gold"]] = (pcs, [cast, ep], target, v)
                        counts["gold"] += 1
                        if counts["gold"] >= POSITIONS_PER_FILE:
                            gold_buf.tofile(
                                os.path.join(golden_dir, f"worker{worker_id}_golden{f_idx['gold']}.bin")
                            )
                            f_idx["gold"] += 1
                            counts["gold"] = 0

                    move_total += 1

                board.push(next_node.move)
                node = next_node

            game_total += 1
            if game_total % 100 == 0:
                queue.put((move_total, game_total))
                move_total, game_total = 0, 0

    if counts["train"] > 0:
        train_buf[:counts["train"]].tofile(os.path.join(train_dir, f"worker{worker_id}_train{f_idx['train']}_end.bin"))
    if counts["gold"] > 0:
        gold_buf[:counts["gold"]].tofile(os.path.join(golden_dir, f"worker{worker_id}_golden{f_idx['gold']}_end.bin"))
    queue.put((move_total, game_total))


def monitor(queue):
    start = time.time()
    total_moves, total_games = 0, 0
    while True:
        msg = queue.get()
        if msg == "DONE":
            break
        move_count, game_count = msg
        total_moves += move_count
        total_games += game_count
        elapsed = time.time() - start
        speed = int(total_moves / elapsed) if elapsed > 0 else 0
        print(
            f"\r[SYNC] Extracted Moves: {total_moves:,} | Games: {total_games:,} | Speed: {speed:,} moves/s",
            end="",
            flush=True,
        )


def run_single_process(train_tasks):
    print(f"\nValidation Complete! Falling back to single-process extraction for {len(train_tasks)} games...")
    worker_process(0, train_tasks, NullQueue())


def run_multi_process(train_tasks):
    print(f"\nValidation Complete! Igniting {WORKERS} Workers for {len(train_tasks)} games...")
    queue = mp.Queue()
    monitor_process = mp.Process(target=monitor, args=(queue,))
    monitor_process.start()

    chunks = split_tasks(train_tasks, WORKERS)
    workers = [mp.Process(target=worker_process, args=(idx, chunk, queue)) for idx, chunk in enumerate(chunks)]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join()

    queue.put("DONE")
    monitor_process.join()


if __name__ == "__main__":
    for directory in [train_dir, val_dir, golden_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    tasks_by_folder = {folder: [] for folder in folders}
    for folder_name in folders:
        path = os.path.join(base_dir, folder_name)
        if not os.path.exists(path):
            continue
        print(f"Scanning folder: {folder_name}...")
        for pgn_path in glob.glob(os.path.join(path, "*.pgn")):
            offsets = fast_scan_pgn_offsets(pgn_path)
            # Make sure BOTH 3500plus and 3600plus are defined as golden/elite
            is_elite = folder_name in ["3500plus", "3600plus"]
            for offset in offsets:
                tasks_by_folder[folder_name].append((pgn_path, offset, is_elite))

    val_tasks = tasks_by_folder["3400plus"][:VAL_GAMES_COUNT]
    train_tasks = (
    tasks_by_folder["3200plus"] [VAL_GAMES_COUNT:]+
    tasks_by_folder["3300plus"] +
    tasks_by_folder["3400plus"] +
    tasks_by_folder["3500plus"] +
    tasks_by_folder["3600plus"] 
)

    print(f"\nMode: Drain Database. Val: {len(val_tasks)} | Train: {len(train_tasks)}")

    print("Extracting Validation on single thread...")
    val_buf = np.zeros(POSITIONS_PER_FILE, dtype=dtype)
    val_index, val_file_index = 0, 0
    for filepath, offset, _ in val_tasks:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as pgn:
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            if not game:
                continue
            board = game.board()
            game_result = game.headers.get("Result", "*")
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                data, _ = get_canonical_data(board, next_node.move, next_node.comment, game_result)
                if data:
                    pcs, cast, ep, target, v, _ = data
                    val_buf[val_index] = (pcs, [cast, ep], target, v)
                    val_index += 1
                    if val_index >= POSITIONS_PER_FILE:
                        val_buf.tofile(os.path.join(val_dir, f"val_{val_file_index}.bin"))
                        val_index = 0
                        val_file_index += 1
                board.push(next_node.move)
                node = next_node
    if val_index > 0:
        val_buf[:val_index].tofile(os.path.join(val_dir, f"val_{val_file_index}_end.bin"))

    if WORKERS <= 1:
        run_single_process(train_tasks)
    else:
        try:
            run_multi_process(train_tasks)
        except (OSError, PermissionError) as exc:
            print(f"\nMultiprocessing unavailable ({exc}).")
            run_single_process(train_tasks)

    # ---------------------------------------------------------
    # NEW: Merge the smaller fragment files into larger block files 
    # ---------------------------------------------------------
    print("\n\nMerging scattered thread fragments to maximize file sizes...")
    merge_fragments(train_dir, "*_end.bin", POSITIONS_PER_FILE, dtype, "train")
    merge_fragments(golden_dir, "*_end.bin", POSITIONS_PER_FILE, dtype, "golden")
    merge_fragments(val_dir, "*_end.bin", POSITIONS_PER_FILE, dtype, "val")

    print("\nInjecting Golden Data (x2)...")
    for bin_path in glob.glob(os.path.join(golden_dir, "*.bin")):
        shutil.copy(bin_path, os.path.join(train_dir, "dup1_" + os.path.basename(bin_path)))
        shutil.copy(bin_path, os.path.join(train_dir, "dup2_" + os.path.basename(bin_path)))

    print("\nExtraction 100% Complete. Database Drained and Ready.")
