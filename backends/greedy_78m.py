from __future__ import annotations

from .common import APP_DIR, GreedyBackend, GreedyConfig


MODEL_PATH = APP_DIR / "models" / "SSChess_78M_BF16.pt"
DEVICE = "cuda"
TOP_MOVES = 3


class SeventyEightMBestMoveBackend(GreedyBackend):
    def __init__(self):
        super().__init__(
            GreedyConfig(
                model_path=MODEL_PATH,
                pick_rank=0,
                device=DEVICE,
                top_moves=TOP_MOVES,
            )
        )
