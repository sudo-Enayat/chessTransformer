# Training loop code.

import os
import warnings
import glob

# --- SILENCE THE NOISE ---
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["NCCL_DEBUG"] = "VERSION"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import time
import zipfile
import gc
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ==========================================
# 1. DDP SETUP
# ==========================================

ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_rank = int(os.environ['RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)

dist.init_process_group(backend='nccl')
master_process = (ddp_rank == 0)

import torch._inductor.config
torch._inductor.config.debug = False

# ==========================================
# 2. AUTOMATIC PATH DISCOVERY
# ==========================================
DATA_DIR = None
PREVIOUS_RUN_DIR = None
resume_ckpt_name = "checkpoint_step_113491.pth"

if master_process:
    print("🔍 Scanning /kaggle/input/ for training data and checkpoints...")

# Walk the mounted directories to find inputs dynamically
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_data" in dirs:
        DATA_DIR = root
    if resume_ckpt_name in files:
        PREVIOUS_RUN_DIR = root

# Fallback defaults if search fails
if DATA_DIR is None:
    DATA_DIR = "/kaggle/input/all-in-one-data"
if PREVIOUS_RUN_DIR is None:
    PREVIOUS_RUN_DIR = "/kaggle/input/34million-chesspositions/SSchess_Checkpoints"

if master_process:
    print(f"✅ Found Data Directory  : {DATA_DIR}")
    print(f"✅ Found Checkpoint Path : {PREVIOUS_RUN_DIR}")

BASE_SAVE_DIR = '/kaggle/working/SSchess_Checkpoints'
TRAINING_START_TIME = time.time()
MAX_RUNTIME_SECONDS = 11.95 * 3600

batch_size = 384
grad_accum_steps = 1 

learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 5000
lr_decay_iters = 105000
max_iters = 120000

eval_interval = 2000
save_interval = 20000
eval_iters = 20

lb_coeff_iters = 2000
dropout_rate = 0.0

n_embd, n_head, n_layer, n_kv_head = 768, 32, 12, 8
num_experts, top_k, router_rate = 1, 1, 1e-9

ptdtype = torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=True)

dist.barrier(device_ids=[ddp_local_rank])

# ==========================================
# 3. DATA LOADER 
# ==========================================
train_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "train_data/*.bin")))[ddp_rank::ddp_world_size]
val_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "val_data/*.bin")))[ddp_rank::ddp_world_size]

# Unified Dtype layout from the prepare_data script
chunk_dtype = np.dtype([
    ('pcs', np.uint8, (64,)), 
    ('meta', np.uint8, (2,)), 
    ('y', np.uint16), 
    ('v', np.int16)
])

class ChunkDataLoader:
    def __init__(self, chunk_files, batch_size, device, seed=1337):
        self.chunk_files = chunk_files
        self.batch_size = batch_size
        self.device = device
        self.current_chunk_idx = 0
        self.rng = np.random.default_rng(seed + ddp_rank)
        self.data = None
        self.pointer = 0
        if self.chunk_files: self.load_chunk(self.chunk_files[0])

    def load_chunk(self, filepath):
        self.data = np.fromfile(filepath, dtype=chunk_dtype)
        chunk_seed = 1337 + ddp_rank + self.current_chunk_idx
        chunk_rng = np.random.default_rng(chunk_seed)
        chunk_rng.shuffle(self.data)
        self.pointer = 0

    def get_batch(self):
        if self.data is None or self.pointer + self.batch_size > len(self.data):
            self.current_chunk_idx = (self.current_chunk_idx + 1) % len(self.chunk_files)
            self.load_chunk(self.chunk_files[self.current_chunk_idx])
        
        batch = self.data[self.pointer : self.pointer + self.batch_size]
        self.pointer += self.batch_size
        
        x_pcs = torch.from_numpy(batch['pcs'].astype(np.int64)).to(self.device, non_blocking=True)
        x_meta = torch.from_numpy(batch['meta'].astype(np.int64)).to(self.device, non_blocking=True)
        y = torch.from_numpy(batch['y'].astype(np.int64)).to(self.device, non_blocking=True)
        v = torch.from_numpy(batch['v'].astype(np.float32) / 10000.0).to(self.device, non_blocking=True)
        return x_pcs, x_meta, y, v

train_loader = ChunkDataLoader(train_chunks, batch_size, device)
val_loader = ChunkDataLoader(val_chunks, batch_size, device)

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
class SpatialGQA(nn.Module):
    def __init__(self, n_head, n_kv_head, head_size):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_size = n_head, n_kv_head, head_size
        self.query = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.key = nn.Linear(n_embd, n_kv_head * head_size, bias=False)
        self.value = nn.Linear(n_embd, n_kv_head * head_size, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.relative_bias_table = nn.Parameter(torch.zeros(15 * 15, n_head))
        coords = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')).flatten(1).T
        rel = coords[None, :, :] - coords[:, None, :] + 7
        self.register_buffer("spatial_index", (rel[:, :, 0] * 15 + rel[:, :, 1]).long(), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        k = torch.repeat_interleave(k, self.n_head // self.n_kv_head, dim=1)
        v = torch.repeat_interleave(v, self.n_head // self.n_kv_head, dim=1)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        attn = attn + self.relative_bias_table[self.spatial_index].permute(2, 0, 1).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (self.attn_dropout(attn) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])

    def forward(self, x):
        out = self.experts[0](x)
        l_aux = torch.tensor(0.0, device=x.device)
        return out, l_aux

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = SpatialGQA(n_head, n_kv_head, n_embd // n_head)
        self.moe = SparseMoE(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x_sa = self.sa(self.ln1(x))
        x = x + x_sa
        x_moe, l_aux = self.moe(self.ln2(x))
        return x + x_moe, l_aux

class SpatialValueHead(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.conv = nn.Conv2d(n_embd, 32, 1)
        self.bn = nn.GroupNorm(8,32)
        self.mlp = nn.Sequential(
            nn.Linear(32*8*8, n_embd), nn.LayerNorm(n_embd), nn.GELU(),
            nn.Linear(n_embd, 1), nn.Tanh()
        )
    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
        x = F.gelu(self.bn(self.conv(x)))
        return self.mlp(x.view(B, -1))

class ChessZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.piece_emb = nn.Embedding(13, n_embd)
        self.pos_emb = nn.Embedding(64, n_embd)
        self.meta_fuse = nn.Linear(n_embd // 4 + n_embd // 2, n_embd)
        self.castling_emb = nn.Embedding(16, n_embd // 4)
        self.ep_emb = nn.Embedding(66, n_embd // 2)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.policy_query = nn.Linear(n_embd, n_embd)
        self.policy_key = nn.Linear(n_embd, n_embd)
        self.value_head = SpatialValueHead(n_embd)

    def forward(self, x_pcs, x_meta, targets=None, values=None, curr_lb_coeff=0.1):
        B, T = x_pcs.shape
        x = self.piece_emb(x_pcs) + self.pos_emb(torch.arange(T, device=x_pcs.device))
        meta = torch.cat([self.castling_emb(x_meta[:,0]), self.ep_emb(x_meta[:,1])], -1)
        x += self.meta_fuse(meta).unsqueeze(1)
        total_aux = 0
        for block in self.blocks:
            x, aux = block(x)
            total_aux += aux
        x = self.ln_f(x)
        policy = torch.bmm(self.policy_query(x), self.policy_key(x).transpose(1,2)) / math.sqrt(n_embd)
        policy = policy.view(B, 4096)
        val = self.value_head(x).squeeze(-1)
        if targets is None: return policy, val, None
        p_loss = F.cross_entropy(policy, targets)
        v_loss = F.mse_loss(val, values)
        loss = p_loss + (curr_lb_coeff * v_loss) + (router_rate * total_aux)
        return policy, val, loss, p_loss, v_loss, total_aux

# ==========================================
# 5. TRAINING HELPERS & INITIALIZATION
# ==========================================

def save_bf16_weights(model, iter, base_dir):
    if master_process:
        os.makedirs(base_dir, exist_ok=True)
        print(f"✨ Exporting clean BF16 weights at step {iter}...")
        raw_model = model._orig_mod.module if hasattr(model, '_orig_mod') else model.module
        raw_sd = raw_model.state_dict()
        clean_sd = {}
        for k, v in raw_sd.items():
            clean_k = k.replace('_orig_mod.', '').replace('module.', '')
            clean_sd[clean_k] = v.to(torch.bfloat16).cpu()
        
        save_path = os.path.join(base_dir, f"SSChess_12B_step_{iter}_bf16.pth")
        torch.save(clean_sd, save_path)
        print(f"✅ BF16 weights saved to: {save_path}")

def save_checkpoint(model, optimizer, train_loader, iter, loss, base_dir):
    if master_process:
        os.makedirs(base_dir, exist_ok=True)
        print(f"\n-> Saving checkpoint at step {iter}...")
        raw_model = model._orig_mod.module if hasattr(model, '_orig_mod') else model.module
        raw_state_dict = raw_model.state_dict()
        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            clean_k = k.replace('_orig_mod.', '').replace('module.', '')
            clean_state_dict[clean_k] = v.cpu()
        
        checkpoint = {
            'model_state_dict': clean_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': iter,
            'loss': loss,
            'chunk_idx': train_loader.current_chunk_idx,
            'pointer': train_loader.pointer
        }
        torch.save(checkpoint, os.path.join(base_dir, f"checkpoint_step_{iter}.pth"))

if master_process: print("🏗️ Initializing model...")

model = ChessZero().to(device)

start_iter = 0
ckpt = None
if resume_ckpt_name:
    ckpt_path = os.path.join(PREVIOUS_RUN_DIR, resume_ckpt_name)
    if not os.path.exists(ckpt_path):
        found = glob.glob(os.path.join(PREVIOUS_RUN_DIR, "**", resume_ckpt_name), recursive=True)
        if found: ckpt_path = found[0]

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_iter = ckpt['iter']
        train_loader.current_chunk_idx = ckpt.get('chunk_idx', 0)
        train_loader.current_chunk_idx = train_loader.current_chunk_idx % len(train_loader.chunk_files)
        train_loader.load_chunk(train_loader.chunk_files[train_loader.current_chunk_idx])
        train_loader.pointer = ckpt.get('pointer', 0)
        if master_process: print(f"-> ✅ Loaded weights from {ckpt_path}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)

if ckpt is not None:
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if master_process: print(f"-> ✅ Loaded optimizer state.")
    del ckpt
    gc.collect()

model = DDP(model, device_ids=[ddp_local_rank])
model = torch.compile(model)

def get_lr(it):
    if it < warmup_iters: return learning_rate * it / warmup_iters
    if it > lr_decay_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (learning_rate - min_lr)

if master_process: print(f"🚀 Training starting at step {start_iter}.")

t0 = time.time()
for iter in range(start_iter, max_iters):
    lb_coeff = 0.5 + (0.5 * min(1.0, iter / lb_coeff_iters))
    lr = get_lr(iter)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    
    if iter % eval_interval == 0 and iter != start_iter:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(eval_iters):
                x_p, x_m, y, v = val_loader.get_batch()
                with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                    _, _, loss, _, _, _ = model(x_p, x_m, y, v, curr_lb_coeff=lb_coeff)
                val_losses.append(loss.item())
        if master_process:
            print(f"\nStep {iter} | Val Loss: {np.mean(val_losses):.4f} | LR: {lr:.2e} | LB: {lb_coeff:.4f}")
        model.train()
        

    if iter > start_iter and iter % save_interval == 0 and master_process:
        save_checkpoint(model, optimizer, train_loader, iter, 0, BASE_SAVE_DIR)
    
        
    optimizer.zero_grad(set_to_none=True)
    batch_loss, batch_p, batch_v, batch_a = 0, 0, 0, 0
    for micro_step in range(grad_accum_steps):
        last_step = (micro_step == grad_accum_steps - 1)
        context = contextlib.nullcontext() if last_step else model.no_sync()
        with context:
            x_p, x_m, y, v = train_loader.get_batch()
            with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
                _, _, loss, p_loss, v_loss, aux_loss = model(x_p, x_m, y, v, curr_lb_coeff=lb_coeff)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            batch_loss += loss.item()
            batch_p += p_loss.item() / grad_accum_steps
            batch_v += v_loss.item() / grad_accum_steps
            batch_a += aux_loss.item() / grad_accum_steps

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    
    if time.time() - TRAINING_START_TIME > MAX_RUNTIME_SECONDS:
        if master_process:
            save_checkpoint(model, optimizer, train_loader, iter, 0, BASE_SAVE_DIR)
        dist.barrier(device_ids=[ddp_local_rank])
        dist.destroy_process_group()
        import sys
        sys.exit(0)
            
    
    if iter % 200 == 0 and master_process:
        dt = time.time() - t0
        t0 = time.time()
        bps = (batch_size * ddp_world_size * grad_accum_steps * 200) / dt
        print(f"iter {iter:5d} | loss {batch_loss:.4f} | {dt*5:.2f}ms/step | P:{batch_p:.2f} V:{batch_v:.2f} A:{batch_a:.2f} | {bps:.0f} b/s | LR: {lr:.1e} | LB: {lb_coeff:.2f}")

if master_process:
    print("Final step reached. Saving final checkpoint...")
    save_checkpoint(model, optimizer, train_loader, max_iters, batch_loss, BASE_SAVE_DIR)
    save_bf16_weights(model, max_iters, BASE_SAVE_DIR)
dist.barrier(device_ids=[ddp_local_rank])

dist.destroy_process_group()
