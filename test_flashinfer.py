import torch
import flashinfer
from utils import *

B, H, S_ori, d = 1, 128, 26, 128
cp_nranks = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
use_log2 = True

torch.manual_seed(42)

q_ori = torch.randn(B, S_ori, H, d, device=device, dtype=dtype)
k_ori = torch.randn(B, S_ori, H, d, device=device, dtype=dtype)
v_ori = torch.randn(B, S_ori, H, d, device=device, dtype=dtype)
scale = 1.0 / math.sqrt(d)

out_ref, lse_ref = attention_causal_with_lse(
    q_ori.transpose(2, 1),
    k_ori.transpose(2, 1),
    v_ori.transpose(2, 1), scale=scale, causal=True, use_log2=use_log2)
out_ref = out_ref.transpose(2, 1)  # (B, S_ori, H, d)
lse_ref = lse_ref.transpose(2, 1)
print(f'{out_ref.dtype=} {lse_ref.dtype=}')

assert q_ori.shape[0] == 1
out_flashinfer, lse_flashinfer = flashinfer.single_prefill_with_kv_cache(
                    q_ori.squeeze(0), k_ori.squeeze(0), v_ori.squeeze(0),
                    causal=True, sm_scale=scale, return_lse=True
                )
out_flashinfer, lse_flashinfer = out_flashinfer.unsqueeze(0), lse_flashinfer.unsqueeze(0)
print(f'{out_flashinfer.dtype=} {lse_flashinfer.dtype=}')
cmp(out_flashinfer, out_ref, "Flashinfer vs Torch")
cmp(lse_flashinfer, lse_ref, "Flashinfer vs Torch (lse)")