from __future__ import annotations

from .common import APP_DIR, MCTSBackend, MCTSConfig


MODEL_PATH = APP_DIR / "models" / "SSChess_78M_BF16.pt"
DEVICE = "cuda"
OPENING_BOOK_FILE = APP_DIR / "engines" / "opening_cache_78m_v3_allroots_cpuct20_1min.scb"

NORMAL_TIME = 5.0
PANIC_TIME = 20.0
CPUCT = 1.0
CPUCT_INIT = 1000.0
CPUCT_SCALE = 0.35
VIRTUAL_LOSS = 0.001
ROOT_NOISE_ALPHA = 0.0
ROOT_NOISE_FRAC = 0.0
EVAL_BATCH_SIZE = 64
REPLICAS = 1
PROGRESS_INTERVAL = 0.5
CACHE_CAPACITY = 2000000
COLLECT_DUP_LIMIT = 64
MIN_SIMS = 256
MAX_SIMS = 50000
USE_FP32 = False
OPENING_BOOK_MB = 0
OPENING_BOOK_MAX_PLY = 8
USE_FP32 = False
OPENING_BOOK_MB = 0
OPENING_BOOK_MAX_PLY = 8
OPENING_BOOK_FULL_PLY = 1
OPENING_BOOK_BRANCHING = 8
OPENING_BOOK_SIMS = 256
OPENING_BOOK_MAX_SECONDS = 3600.0


class SeventyEightMMCTSBackend(MCTSBackend):
    def __init__(self):
        super().__init__(
            MCTSConfig(
                model_path=MODEL_PATH,
                device=DEVICE,
                normal_time=NORMAL_TIME,
                panic_time=PANIC_TIME,
                cpuct=CPUCT,
                cpuct_init=CPUCT_INIT,
                cpuct_scale=CPUCT_SCALE,
                virtual_loss=VIRTUAL_LOSS,
                root_noise_alpha=ROOT_NOISE_ALPHA,
                root_noise_frac=ROOT_NOISE_FRAC,
                eval_batch_size=EVAL_BATCH_SIZE,
                replicas=REPLICAS,
                progress_interval=PROGRESS_INTERVAL,
                cache_capacity=CACHE_CAPACITY,
                collect_dup_limit=COLLECT_DUP_LIMIT,
                min_sims=MIN_SIMS,
                max_sims=MAX_SIMS,
                use_fp32=USE_FP32,
                opening_book_file=OPENING_BOOK_FILE,
                opening_book_mb=OPENING_BOOK_MB,
                opening_book_max_ply=OPENING_BOOK_MAX_PLY,
                opening_book_full_ply=OPENING_BOOK_FULL_PLY,
                opening_book_branching=OPENING_BOOK_BRANCHING,
                opening_book_sims=OPENING_BOOK_SIMS,
                opening_book_max_seconds=OPENING_BOOK_MAX_SECONDS,
            )
        )


class SeventyEightMMCTSRefereeBackend(MCTSBackend):
    def __init__(self):
        super().__init__(
            MCTSConfig(
                model_path=MODEL_PATH,
                device=DEVICE,
                normal_time=3600.0,
                panic_time=3600.0,
                cpuct=1.0,
                cpuct_init=1000.0,
                cpuct_scale=0.35,
                virtual_loss=0.001,
                root_noise_alpha=0.0,
                root_noise_frac=0.0,
                eval_batch_size=64,
                replicas=1,
                progress_interval=0.5,
                cache_capacity=2000000,
                collect_dup_limit=64,
                min_sims=256,
                max_sims=256,
                use_fp32=False,
                opening_book_file=OPENING_BOOK_FILE,
                opening_book_mb=0,
                opening_book_max_ply=8,
                opening_book_full_ply=1,
                opening_book_branching=8,
                opening_book_sims=256,
                opening_book_max_seconds=3600.0,
            )
        )
