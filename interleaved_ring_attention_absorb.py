import sys
import torch
from utils import *
    
def main():
    torch.manual_seed(42)
    
    B, S_ori = 1, 26
    cp_nranks = 4
    
    h_q, h_kv= 128, 1
    kv_lora_rank = 512
    v_head_dim = 128
    qk_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    
    assert B == 1, "This test only supports B=1"
    
    # 确保序列长度能被 rank 数整除
    S = (S_ori + cp_nranks - 1) // cp_nranks * cp_nranks
    S_local = S // cp_nranks
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    use_log2 = True
    merge_after_absorb = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True
    print(f'{merge_after_absorb=}')
    
    # 定义统一的缩放因子
    scale = (qk_head_dim + qk_rope_head_dim) ** -0.5
    print(f"Using manual scale: {scale}")
    
    # print all parameters
    print(f"B={B}, h_q={h_q}, h_kv={h_kv}, S_ori={S_ori} (Padded S={S}), "
          f"qk_head_dim={qk_head_dim}, qk_rope_head_dim={qk_rope_head_dim}, "
          f"kv_lora_rank={kv_lora_rank}, v_head_dim={v_head_dim}, CP={cp_nranks}")
    print(f"Using device: {device}, dtype: {dtype}, use_log2: {use_log2}")
    
    q_nope_ori = torch.randn(B, S_ori, h_q, qk_head_dim, device=device, dtype=dtype)
    q_pe_ori = torch.randn(B, S_ori, h_q, qk_rope_head_dim, device=device, dtype=dtype)
    c_kv_ori = torch.randn(B, S_ori, kv_lora_rank, device=device, dtype=dtype)
    k_pe_ori = torch.randn(B, S_ori, qk_rope_head_dim, device=device, dtype=dtype)
    
    wkv_b_q = torch.randn(h_q, kv_lora_rank, qk_head_dim, device=device, dtype=dtype)
    wkv_b_o = torch.randn(h_q, kv_lora_rank, v_head_dim, device=device, dtype=dtype)
    
    # padding zero
    q_nope = torch.zeros(B, S, h_q, qk_head_dim, device=device, dtype=dtype)
    q_pe = torch.zeros(B, S, h_q, qk_rope_head_dim, device=device, dtype=dtype)
    c_kv = torch.zeros(B, S, kv_lora_rank, device=device, dtype=dtype)
    k_pe = torch.zeros(B, S, qk_rope_head_dim, device=device, dtype=dtype)
    
    q_nope[:, :S_ori, :, :] = q_nope_ori
    q_pe[:, :S_ori, :, :] = q_pe_ori
    c_kv[:, :S_ori, :] = c_kv_ori
    k_pe[:, :S_ori, :] = k_pe_ori
    
    print("Computing Non absorb...")
    q = torch.concat([q_nope, q_pe], dim=3)  # (B, S, h_q, qk_head_dim + qk_rope_head_dim)
    c_kv_broadcast = c_kv.unsqueeze(-2).expand(-1, -1, h_q, -1).unsqueeze(-2)  # (B, S, h_q, 1, kv_lora_rank)
    k_nope = (c_kv_broadcast @ wkv_b_q).squeeze(-2)  # (B, S, h_q, qk_head_dim)
    
    k = torch.concat([k_nope, k_pe.unsqueeze(-2).expand(-1, -1, h_q, -1)], dim=3)  # (B, S, h_q, qk_head_dim + qk_rope_head_dim)
    v = (c_kv_broadcast @ wkv_b_o).squeeze(-2)  # (B, S, h_q, qk_head_dim)
    
    print(f'{q.shape=} {k.shape=} {v.shape=}')
    out_ref, lse_ref = attention_causal_with_lse(
        q.transpose(2, 1),
        k.transpose(2, 1),
        v.transpose(2, 1), scale=scale, causal=True, use_log2=use_log2)
    out_ref = out_ref.transpose(2, 1)  # (B, S_ori, H, d)
    print(f'{out_ref.shape=} {lse_ref.shape=}')

    print("Computing absorb...")
    q_absorb = (q_nope.unsqueeze(-2) @ wkv_b_q.transpose(-1, -2)).squeeze(-2)  # (B, S, h_q, kv_lora_rank)
    q2 = torch.concat([q_absorb, q_pe], dim=3)  # (B, S, h_q, kv_lora_rank + qk_rope_head)
    
    k2_concat = torch.concat([c_kv, k_pe], dim=-1) # (B, S, 576)
    k2 = k2_concat.unsqueeze(-2).expand(-1, -1, h_q, -1).clone()  # (B, S, h_q, kv_lora_rank + qk_rope_head)
    v2 = c_kv.unsqueeze(-2).expand(-1, -1, h_q, -1).clone()  # (B, S, h_q, kv_lora_rank)
    
    print(f'{q2.shape=} {k2.shape=} {v2.shape=}')
    out_ref2_raw, lse_ref2 = attention_causal_with_lse(
        q2.transpose(2, 1),
        k2.transpose(2, 1),
        v2.transpose(2, 1), scale=scale, causal=True, use_log2=use_log2)
    out_ref2_raw = out_ref2_raw.transpose(2, 1)
    out_ref2 = (out_ref2_raw.unsqueeze(-2) @ wkv_b_o).squeeze(-2)
    print(f'{out_ref2.shape=} {lse_ref2.shape=}')
    
    # compare
    cmp(out_ref2, out_ref, "Torch Absorb vs Non-Absorb")
    
    print("Computing interleaved ring attention...")
    q_dist = q2.reshape(B, S_local, cp_nranks, h_q, kv_lora_rank + qk_rope_head_dim)
    c_kv_dist = c_kv.reshape(B, S_local, cp_nranks, kv_lora_rank)
    k_pe_dist = k_pe.reshape(B, S_local, cp_nranks, qk_rope_head_dim)
    out_ring_dist = torch.zeros((B, S_local, cp_nranks, h_q, v_head_dim), device=device, dtype=dtype)  # (B, S_local, CP, H, d)
    for i in range(cp_nranks):
        q_local = q_dist[:, :, i, :, :].transpose(1, 2) # (B, h_q, S_local, kv_lora_rank + qk_rope_head_dim)
        
        acc_out = None  # (B, H, S_local, d)  # float32
        acc_lse = None  # (B, H, S_local)
        for step in range(cp_nranks):
            j = (i - step + cp_nranks) % cp_nranks
            
            c_kv_j = c_kv_dist[:, :, j, :] # (B, S_local, kv_lora_rank)
            k_pe_j = k_pe_dist[:, :, j, :]  # (B, S_local, qk_rope_head_dim)
            k_concat_j = torch.concat([c_kv_j, k_pe_j], dim=-1)  # (B, S_local, kv_lora_rank + qk_rope_head_dim)
            
            k_local = k_concat_j.unsqueeze(-2).expand(-1, -1, h_q, -1).transpose(1, 2)  # (B, h_q, S_local, kv_lora_rank + qk_rope_head)
            v_local = c_kv_j.unsqueeze(-2).expand(-1, -1, h_q, -1).transpose(1, 2)  # (B, h_q, S_local, kv_lora_rank)
            
            if i >= j:
                # with full triangular mask
                block_out, block_lse = attention_causal_with_lse(
                    q_local, k_local, v_local, scale=scale, causal=True, use_log2=use_log2
                )
            else:
                # with sub-triangular mask (diagonal removed)
                block_out_sub, block_lse_sub = attention_causal_with_lse(
                    q_local[:, :, 1:, :],
                    k_local[:, :, :-1, :], v_local[:, :, :-1, :],
                    scale=scale, causal=True, use_log2=use_log2
                )
                
                block_out = torch.zeros((B, h_q, S_local, kv_lora_rank), device=device, dtype=torch.float32)
                block_lse = torch.full((B, h_q, S_local), float('-inf'), device=device, dtype=torch.float32)
                
                block_out[:, :, 1:, :] = block_out_sub
                block_lse[:, :, 1:] = block_lse_sub
            
            # print(f'{block_out.shape=}, {block_lse.shape=}')
            if merge_after_absorb:
                block_out = (block_out.transpose(1, 2).unsqueeze(-2) @ wkv_b_o).squeeze(-2).transpose(1, 2)  # (B, h_q, S_local, v_head_dim)
                # print(f'Absorb: {block_out.shape=}, {block_lse.shape=}')
            
            acc_out, acc_lse = update_out_and_lse(acc_out, acc_lse, block_out, block_lse, use_log2)
            # print(f'{acc_out.shape=}')
            
        if not merge_after_absorb:
            acc_out = (acc_out.transpose(1, 2).unsqueeze(-2) @ wkv_b_o).squeeze(-2).transpose(1, 2)  # (B, h_q, S_local, v_head_dim)
        out_ring_dist[:, :, i, :, :] = acc_out.transpose(1, 2).to(dtype)

    out_ring = out_ring_dist.view(B, S, h_q, v_head_dim)
    print(f'{out_ring.shape=} {out_ring.dtype=}')
    
    print("\nFinal Comparison:")
    cmp(out_ring[:, :S_ori, :, :], out_ref[:, :S_ori, :, :], "Interleaved Ring vs Torch Global")
    
    print(f'{torch.allclose(out_ring, out_ref, atol=1e-1, rtol=1e-6)=}')

if __name__ == "__main__":
    main()