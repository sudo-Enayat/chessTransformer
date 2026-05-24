# ChessTransformer

Warning: an NVIDIA GPU is required to run this app. The included models are CUDA-bound TorchScript exports and will not run on CPU-only systems.

Web-playable chess UI with:

- `Vs Friend`
- `Vs AI` difficulty tiers
- 12M and 78M greedy backends
- 12M and 78M native MCTS backends
- board themes, move history, captured pieces, eval bar, prediction, undo, resign, and review tools

The 78M backend defaults to the 80k BF16 TorchScript export for faster native MCTS inference.

## Quick Setup (Recommended)

If you received this as a zip file, just run the one-click setup script:

```powershell
.\setup.ps1
```

This will verify your system, create a virtual environment, install all dependencies, and launch the app automatically. You only need **Python 3.12** and an **NVIDIA GPU** installed beforehand.

## Requirements

- Windows
- Python 3.12
- NVIDIA GPU with CUDA support
- Git LFS installed

## Repository Notes

- `models/SSChess_12M.pt`, `models/SSChess_78M_BF16.pt`, and `models/SSChess_78M_FP32.pt` are tracked with Git LFS.
- The native MCTS engine binary is included at `engines/native_mcts.exe`.
- The app is self-contained inside this folder and no longer depends on a sibling `SSChess_training` directory.

## Setup

1. Clone the repository with Git LFS enabled:

```powershell
git lfs install
git clone <repo-url>
cd chessTransformer
```

2. Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
.\run_web_app.ps1
```

Or:

```powershell
.\.venv\Scripts\python.exe .\web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Included Files

- `web_app.py`: Flask server
- `templates/index.html`: web UI
- `backends/`: greedy and MCTS backend implementations
- `models/`: 12M and 78M TorchScript checkpoints
- `engines/native_mcts.exe`: native MCTS search engine

## Troubleshooting

- If model loading fails, verify that your GPU and driver support CUDA 12.1.
- If Git LFS was not installed before clone, run `git lfs pull` after installing it.
- If Windows blocks the native executable, unblock it from file properties or PowerShell execution prompts.
