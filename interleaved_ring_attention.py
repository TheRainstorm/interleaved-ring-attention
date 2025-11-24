import torch
from utils import *

def main():
    torch.manual_seed(42)
    
    # B: 批处理大小 (Batch size)
    # H: 注意力头数 (Number of heads)
    # S: 序列长度 (Sequence length)
    # d: 维度 (Dimension)
    B, H, S_ori, d = 1, 128, 26, 128 # 减小尺寸以便快速测试，逻辑与大尺寸一致
    cp_nranks = 4
    
    # 确保序列长度能被 rank 数整除
    S = (S_ori + cp_nranks - 1) // cp_nranks * cp_nranks
    S_local = S // cp_nranks
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    use_log2 = True # 所有 exp 和 log 都使用 base-2
    
    print(f"B={B}, H={H}, S={S_ori} (Local S={S_local}), d={d}, CP={cp_nranks}")
    print(f"Using device: {device}, dtype: {dtype}, use_log2: {use_log2}")
    
    q_ori = torch.randn(B, S_ori, H, d, device=device, dtype=dtype)
    k_ori = torch.randn(B, S_ori, H, d, device=device, dtype=dtype)
    v_ori = torch.randn(B, S_ori, H, d, device=device, dtype=dtype)
    # padding zero
    q = torch.zeros(B, S, H, d, device=device, dtype=dtype)
    k = torch.zeros(B, S, H, d, device=device, dtype=dtype)
    v = torch.zeros(B, S, H, d, device=device, dtype=dtype)
    q[:, :S_ori, :, :] = q_ori
    k[:, :S_ori, :, :] = k_ori
    v[:, :S_ori, :, :] = v_ori
    
    scale = 1.0 / math.sqrt(d)

    print("Computing global reference attention...")
    out_ref, lse_ref = attention_causal_with_lse(
        q_ori.transpose(2, 1),
        k_ori.transpose(2, 1),
        v_ori.transpose(2, 1), scale=scale, causal=True, use_log2=use_log2)
    out_ref = out_ref.transpose(2, 1)  # (B, S_ori, H, d)
    print(f'{out_ref.dtype=} {lse_ref.dtype=}')

    print("Computing interleaved ring attention...")
    
    q_dist = q.reshape(B, S_local, cp_nranks, H, d)
    k_dist = k.reshape(B, S_local, cp_nranks, H, d)
    v_dist = v.reshape(B, S_local, cp_nranks, H, d)
    ring_final_out = torch.zeros_like(q_dist)  # (B, S_local, CP, H, d)
    print(f'{q_dist.dtype=} {ring_final_out.dtype=}')
    for i in range(cp_nranks):
        q_local = q_dist[:, :, i, :, :].transpose(1, 2)
        
        acc_out = None  # (B, H, S_local, d)
        acc_lse = None  # (B, H, S_local)
        for step in range(cp_nranks):
            j = (i - step + cp_nranks) % cp_nranks
            
            k_local = k_dist[:, :, j, :, :].transpose(1, 2)
            v_local = v_dist[:, :, j, :, :].transpose(1, 2)
            
            if i >= j:
                block_out, block_lse = attention_causal_with_lse(
                    q_local, k_local, v_local, scale=scale, causal=True, use_log2=use_log2
                )
            else:
                block_out_sub, block_lse_sub = attention_causal_with_lse(
                    q_local[:, :, 1:, :],
                    k_local[:, :, :-1, :], v_local[:, :, :-1, :],
                    scale=scale, causal=True, use_log2=use_log2
                )
                
                block_out = torch.zeros((B, H, S_local, d), device=device, dtype=dtype)
                block_lse = torch.zeros((B, H, S_local), device=device, dtype=dtype)
                # trick: copy first
                block_out[:, :, 0, :] = acc_out[:, :, 0, :]
                block_lse[:, :, 0] = acc_lse[:, :, 0]
                
                block_out[:, :, 1:, :] = block_out_sub
                block_lse[:, :, 1:] = block_lse_sub
                
            acc_out, acc_lse = update_out_and_lse(acc_out, acc_lse, block_out, block_lse, use_log2)
        
        ring_final_out[:, :, i, :, :] = acc_out.transpose(1, 2)

    out_ring = ring_final_out.view(B, S, H, d)[:, :S_ori, :, :]
    
    print("\nFinal Comparison:")
    cmp(out_ring, out_ref, "Interleaved Ring vs Torch Global")
    
    assert torch.allclose(out_ring, out_ref, atol=1e-4, rtol=1e-4), "Verification Failed!"
    print("SUCCESS: Algorithm verified.")

if __name__ == "__main__":
    main()