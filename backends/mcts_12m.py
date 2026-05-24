from __future__ import annotations

from .common import APP_DIR, MCTSBackend, MCTSConfig


MODEL_PATH = APP_DIR / "models" / "SSChess_12M.pt"
DEVICE = "cuda"

NORMAL_TIME = 5.0
PANIC_TIME = 20.0
CPUCT = 1.0
CPUCT_INIT = 1000.0
CPUCT_SCALE = 0.35
VIRTUAL_LOSS = 0.001
ROOT_NOISE_ALPHA = 0.0
ROOT_NOISE_FRAC = 0.0
EVAL_BATCH_SIZE = 768
REPLICAS = 1
PROGRESS_INTERVAL = 0.5
CACHE_CAPACITY = 50000
COLLECT_DUP_LIMIT = 128
MIN_SIMS = 4000
MAX_SIMS = 0
USE_FP32 = False


class TwelveMMCTSBackend(MCTSBackend):
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
            )
        )
