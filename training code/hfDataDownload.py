# Directly download stockfish data of midgames

import sys
import subprocess
import os
import socket

# Set a strict 45-second connection timeout to prevent any HF download from hanging forever
socket.setdefaulttimeout(45)

# =====================================================================
# PART 1: AUTO-INSTALL MISSING PACKAGES
# =====================================================================
try:
    import chess
    import chess.pgn
except ImportError:
    print("📦 python-chess is missing. Installing automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-chess"])
    import chess
    import chess.pgn

try:
    import huggingface_hub
except ImportError:
    print("📦 huggingface_hub is missing. Installing automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

try:
    import gdown
except ImportError:
    print("📦 gdown is missing. Installing automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    import gdown

# =====================================================================
# PART 2: DOWNLOAD VALIDATION DATA FROM GDRIVE
# =====================================================================
base_dir = "/kaggle/working"
val_dir = os.path.join(base_dir, "val_data")
os.makedirs(val_dir, exist_ok=True)

# Only download if the folder is empty or doesn't exist
if len(os.listdir(val_dir)) == 0:
    print("🧱 Downloading validation data from Google Drive...")
    try:
        # Downloads your val_0.bin and val_merged_end.bin folder
        subprocess.check_call([
            "gdown", "--folder", "1WH4P3lx1KQzlXP_Ls-ZyjExqGd5dsYFj", "-O", val_dir, "--remaining-ok"
        ])
        print("✅ Validation data downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download validation data: {e}")
else:
    print("✅ Validation data already exists. Skipping download.")

# =====================================================================
# PART 3: THE SEQUENTIAL TRAINING EXTRACTION PIPELINE
# =====================================================================
import gzip
import shutil
import glob
import math
import re
import time
import multiprocessing as mp
import numpy as np
from huggingface_hub import list_repo_files, hf_hub_download

train_dir = os.path.join(base_dir, "train_data")
os.makedirs(train_dir, exist_ok=True)

# Extraction Constants
WORKERS = 4
POSITIONS_PER_FILE = 750000
TARGET_TOTAL_FILES = 24  # Stop at exactly 24 files (1.26 GB)

dtype = [("pcs", np.uint8, (64,)), ("meta", np.uint8, (2,)), ("y", np.uint16), ("v", np.int16)]

FISHTEST_REGEX = re.compile(r"^([^\s/]+)/[0-9]+")
BRACKET_EVAL_REGEX = re.compile(r"\[%eval\s+([^\]]+)\]", re.IGNORECASE)
PLAIN_EVAL_REGEX = re.compile(r"(?:^|[\s\)])([+\-]?(?:[Mm]\d+|#\d+|\d+(?:\.\d+)?)|[Mm][+\-]?\d+)\s*/\s*\d+", re.IGNORECASE)
HASH_MATE_REGEX = re.compile(r"([+\-]?#[0-9]+)", re.IGNORECASE)
OPENING_REGEX = re.compile(r"([A-E]\d{2})[:\s]")
DRAW_REGEX = re.compile(r"\bDRAW\b", re.IGNORECASE)
WHITE_WIN_REGEX = re.compile(r"\bWHITE WINS\b", re.IGNORECASE)
BLACK_WIN_REGEX = re.compile(r"\bBLACK WINS\b", re.IGNORECASE)

def result_to_cp(result):
    if result == "1-0": return 15000
    if result == "0-1": return -15000
    if result == "1/2-1/2": return 0
    return None

def parse_eval_token(token, is_percent_eval=False):
    token = token.strip()
    if not token: return None, False
    token_upper = token.upper()
    if token_upper.startswith("M+") or token_upper.startswith("M-"):
        return (15000 if token_upper[1] == "+" else -15000), True
    if token_upper.startswith(("+M", "-M", "M")) or "#" in token_upper:
        return (15000 if not token.startswith("-") else -15000), True
    try:
        numeric = float(token)
    except ValueError:
        return None, False
    cp = int(round(numeric if (is_percent_eval and "." not in token) else numeric * 100))
    cp = max(min(cp, 15000), -15000)
    return cp, abs(cp) >= 15000

def extract_comment_value(comment, board_turn, game_result=None):
    if not comment: return None, False, False
    comment = comment.strip()
    if not comment: return None, False, False
    comment_upper = comment.upper()
    
    if "BOOK" in comment_upper or OPENING_REGEX.search(comment_upper):
        return 0, True, False

    fish_match = FISHTEST_REGEX.search(comment_upper)
    if fish_match:
        val_str = fish_match.group(1)
        if "#" in val_str or "M" in val_str:
            val = -15000 if "-" in val_str else 15000
            return val, False, True
        else:
            try:
                cp_val = int(round(float(val_str) * 100))
                cp_val = max(min(cp_val, 15000), -15000)
                return cp_val, False, abs(cp_val) >= 15000
            except ValueError:
                pass

    bracket_match = BRACKET_EVAL_REGEX.search(comment)
    if bracket_match:
        eval_token = bracket_match.group(1).split(",", 1)[0]
        val, is_mate = parse_eval_token(eval_token, is_percent_eval=True)
        if val is not None: return val, False, is_mate

    result_cp = result_to_cp(game_result)
    if result_cp is not None:
        if DRAW_REGEX.search(comment_upper) or WHITE_WIN_REGEX.search(comment_upper) or BLACK_WIN_REGEX.search(comment_upper) or "ADJUDICATION" in comment_upper:
            if board_turn == chess.BLACK: result_cp = -result_cp
            return result_cp, False, abs(result_cp) >= 15000
    return None, False, False

def get_canonical_data(board, move, comment, game_result):
    val, is_book, is_mate = extract_comment_value(comment, board.turn, game_result=game_result)
    if val is None: return None, False
    wdl_val = math.tanh(val / 400.0)
    v_target = int(wdl_val * 10000)
    pcs = np.zeros(64, dtype=np.uint8)
    flip = board.turn == chess.BLACK
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece: continue
        piece_color = (not piece.color) if flip else piece.color
        piece_type = piece.piece_type + (6 if piece_color == chess.BLACK else 0)
        mapped_sq = sq ^ 0x38 if flip else sq
        pcs[mapped_sq] = piece_type
    if not flip:
        cast = ((1 if board.has_kingside_castling_rights(chess.WHITE) else 0) | (2 if board.has_queenside_castling_rights(chess.WHITE) else 0) | (4 if board.has_kingside_castling_rights(chess.BLACK) else 0) | (8 if board.has_queenside_castling_rights(chess.BLACK) else 0))
        ep = board.ep_square if board.ep_square is not None else 64
    else:
        cast = ((1 if board.has_kingside_castling_rights(chess.BLACK) else 0) | (2 if board.has_queenside_castling_rights(chess.BLACK) else 0) | (4 if board.has_kingside_castling_rights(chess.WHITE) else 0) | (8 if board.has_queenside_castling_rights(chess.WHITE) else 0))
        ep = (board.ep_square ^ 0x38) if board.ep_square is not None else 64
    from_sq, to_sq = move.from_square, move.to_square
    if flip:
        from_sq ^= 0x38
        to_sq ^= 0x38
    y_target = np.uint16((from_sq * 64) + to_sq)
    return (pcs, cast, ep, y_target, v_target, is_mate), is_book

def fast_scan_pgn_offsets(filepath):
    offsets = []
    with open(filepath, "rb") as handle:
        offset = 0
        for line in handle:
            if line.startswith(b"[Event "): offsets.append(offset)
            offset += len(line)
    return offsets

def split_tasks(tasks, num_chunks):
    if not tasks: return []
    if num_chunks <= 1: return [tasks]
    chunk_size = math.ceil(len(tasks) / num_chunks)
    return [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

def worker_process(worker_id, tasks, queue, file_step):
    train_buf = np.zeros(POSITIONS_PER_FILE, dtype=dtype)
    counts = {"train": 0}
    f_idx = {"train": file_step} # Align indexing dynamically to prevent overwriting
    move_total, game_total = 0, 0
    
    for filepath, offset in tasks:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as pgn:
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            if not game: continue
            board = game.board()
            game_result = game.headers.get("Result", "*")
            node = game
            while node and not node.is_end() and node.variations:
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
                    move_total += 1
                board.push(next_node.move)
                node = next_node
                
            game_total += 1
            if game_total % 100 == 0:
                queue.put((move_total, game_total))
                move_total, game_total = 0, 0
                
    if counts["train"] > 0:
        train_buf[:counts["train"]].tofile(os.path.join(train_dir, f"worker{worker_id}_train{f_idx['train']}_end.bin"))
    queue.put((move_total, game_total))

def count_completed_full_files():
    full_files = [f for f in glob.glob(os.path.join(train_dir, "*.bin")) if not f.endswith("_end.bin")]
    return len(full_files)

def monitor(queue):
    start = time.time()
    total_moves, total_games = 0, 0
    while True:
        msg = queue.get()
        if msg == "DONE": break
        move_count, game_count = msg
        total_moves += move_count
        total_games += game_count
        speed = int(total_moves / (time.time() - start)) if (time.time() - start) > 0 else 0
        print(f"\r[SYNC] Extracted Moves: {total_moves:,} | Games: {total_games:,} | Speed: {speed:,} moves/s", end="", flush=True)

if __name__ == "__main__":
    print("🔍 Fetching latest dataset files from Hugging Face...")
    all_repo_files = list_repo_files(repo_id="official-stockfish/fishtest_pgns", repo_type="dataset")
    pgn_files = [f for f in all_repo_files if f.endswith(".pgn.gz")]

    # Get the latest files
    target_files = sorted([f for f in pgn_files if f.startswith("26-")], reverse=True)
    if len(target_files) < 40:
        late_2025_files = sorted([f for f in pgn_files if f.startswith("25-")], reverse=True)
        target_files = target_files + late_2025_files

    print(f"🎬 Sequential pipeline initialized. Target: {TARGET_TOTAL_FILES} full files (~1.26 GB)")
    
    file_step = 0
    while count_completed_full_files() < TARGET_TOTAL_FILES and file_step < len(target_files):
        current_full_count = count_completed_full_files()
        print(f"\n⚡ Progress: {current_full_count}/{TARGET_TOTAL_FILES} full files.")
        
        hf_filename = target_files[file_step]
        print(f"📥 Downloading: {hf_filename}")
        
        try:
            # Download file
            local_gz_path = hf_hub_download(
                repo_id="official-stockfish/fishtest_pgns", 
                repo_type="dataset", 
                filename=hf_filename
            )
        except Exception as e:
            # If the download times out or fails, skip immediately to prevent hanging
            print(f"⚠️ Connection error or timeout on {hf_filename}. Skipping to next file...")
            file_step += 1
            continue
        
        # Unzip to a temporary single file
        temp_pgn_path = os.path.join(base_dir, "temp_parsing.pgn")
        with gzip.open(local_gz_path, 'rb') as f_in:
            with open(temp_pgn_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Scan offsets of this single file
        offsets = fast_scan_pgn_offsets(temp_pgn_path)
        train_tasks = [(temp_pgn_path, offset) for offset in offsets]

        # Extract train sequentially using multi-threading
        queue = mp.Queue()
        monitor_process = mp.Process(target=monitor, args=(queue,))
        monitor_process.start()
        
        chunks = split_tasks(train_tasks, WORKERS)
        workers = [mp.Process(target=worker_process, args=(idx, chunk, queue, file_step)) for idx, chunk in enumerate(chunks)]
        for proc in workers: proc.start()
        for proc in workers: proc.join()
        
        queue.put("DONE")
        monitor_process.join()
        
        # IMMEDIATELY Delete the temporary raw PGN to keep disk space minimal
        if os.path.exists(temp_pgn_path):
            os.remove(temp_pgn_path)
            
        file_step += 1

    # Cleanup incomplete partial _end.bin files
    print("\n🧹 Final cleanup of fractional end-files...")
    for partial_file in glob.glob(os.path.join(train_dir, "*_end.bin")):
        os.remove(partial_file)
        
    print(f"\n🎉 Extraction complete! Exact 24 files generated successfully. Clean workspace.")
