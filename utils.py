import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math

def cmp(o1, o2, msg='cmp'):
    """比较两个 Tensor 的相似度"""
    if o1.shape != o2.shape:
        print(f"{msg}: Shape mismatch! {o1.shape} vs {o2.shape}")
        return
    
    # 移动到 CPU 并转为 float32 避免精度问题造成的误报
    o1 = o1.float().cpu()
    o2 = o2.float().cpu()
    
    max_error = torch.max(torch.abs(o1 - o2)).item()
    mean_abs_error = (torch.sum(torch.abs(o1 - o2)) / o2.numel()).item()

    positions = (torch.abs(o1 - o2) == max_error).nonzero(as_tuple=False)
    first_position = tuple(positions[0].tolist())
    value1 = o1[first_position].item()
    value2 = o2[first_position].item()
    
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(o1.flatten(), o2.flatten(), dim=0).item()
    
    print(f'{msg:35s}: CosSim={cos_sim:.6f}, MAE={mean_abs_error:.6f}, MaxError={max_error:.6f} ({value1:.6f} {value2:.6f})')

def attention_causal_with_lse(query_states, key_states, value_states,
                                  scale=None, causal=True, use_log2=False):
    """
    计算单块内的因果注意力，支持 GQA (Grouped-Query Attention)。

    - 当 Q, K, V 的头数量相同时，为标准的多头注意力 (MHA)。
    - 当 K, V 只有 1 个头时，为多查询注意力 (MQA)。
    - 当 Q 的头数量是 K, V 的整数倍时，为分组查询注意力 (GQA)。

    Input Shapes:
    - query_states: (B, H_q, S, d)  (B: batch_size, H_q: num_query_heads, S: seq_len, d: head_dim)
    - key_states:   (B, H_k, S, d)  (H_k: num_key_value_heads)
    - value_states: (B, H_k, S, d)

    Output:
    - out: 注意力输出，形状为 (B, H_q, S, d)，始终是全 base-e 的。
    - lse: Log-Sum-Exp，形状为 (B, H_q, S)。当 use_log2=True 时返回 lse/ln2。
    """
    B, H_q, S, d = query_states.shape
    _, H_k, _, _ = key_states.shape

    # 如果是 GQA 或 MQA，需要重复 K 和 V 的头来匹配 Q
    if H_q != H_k:
        # 确认 H_q 是 H_k 的整数倍
        if H_q % H_k != 0:
            raise ValueError(f"GQA requires num_query_heads ({H_q}) to be a multiple of num_key_value_heads ({H_k})")
        
        num_reps = H_q // H_k
        print(f'{num_reps=} {key_states.shape=} {query_states.shape=}')
        # 沿头的维度重复 K 和 V
        # (B, H_k, S, d) -> (B, H_k, 1, S, d) -> (B, H_k, num_reps, S, d) -> (B, H_q, S, d)
        key_states = key_states.unsqueeze(2).expand(-1, -1, num_reps, -1, -1).reshape(B, H_q, S, d)
        value_states = value_states.unsqueeze(2).expand(-1, -1, num_reps, -1, -1).reshape(B, H_q, S, d)
        # 更高效的写法是使用 repeat_interleave
        # key_states = key_states.repeat_interleave(num_reps, dim=1)
        # value_states = value_states.repeat_interleave(num_reps, dim=1)

    if scale is None:
        scale = d ** -0.5
    
    # 经过 GQA 处理后，Q, K, V 的头数量都为 H_q，后续计算与 MHA 相同
    attn_scores = query_states @ key_states.transpose(-2, -1)  # (B, H_q, S, S)
    attn_scores = attn_scores.float()
    attn_scores = attn_scores * scale
    
    if causal:
        mask = torch.triu(torch.ones(S, S, device=query_states.device, dtype=torch.bool), diagonal=1)
        attn_scores.masked_fill_(mask, float('-inf'))
        
    if not use_log2:
        # 1. 标准 LSE (base e)
        lse = torch.logsumexp(attn_scores, dim=-1)  # (B, H_q, S)
    else:
        # 2. Base-2 LSE
        ln_2 = math.log(2.0)
        # 参考 FlashAttention/Flash-MLA，实际计算的是 log2(sum(e^x))
        lse = torch.logsumexp(attn_scores, dim=-1) / ln_2
    
    attn_weights = attn_scores.softmax(dim=-1)
    attn_weights = attn_weights.to(query_states.dtype)
    out = attn_weights @ value_states  # (B, H_q, S, d)
    
    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],  # (B, H, S_local, d)
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    use_log2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if out is None:
        return block_out.float(), block_lse
    
    return _update_out_and_lse_impl(out, lse, block_out.float(), block_lse, use_log2)

def _update_out_and_lse_impl(
    out: torch.Tensor,          # all float32
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    use_log2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在线更新 Ring Attention 的累积结果 (Online Softmax 逻辑)
    """
    lse, block_lse = lse.unsqueeze(dim=-1), block_lse.unsqueeze(dim=-1) #(B, H, S, 1)

    if not use_log2:
        # ref: https://github.com/zhuzilin/ring-flash-attention
        # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
        new_out = out - F.sigmoid(block_lse - lse) * (out - block_out)
        new_lse = lse - F.logsigmoid(lse - block_lse)

        return new_out, new_lse.squeeze(-1)

    log, exp = torch.log, torch.exp
    if use_log2:
        log, exp = torch.log2, torch.exp2

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For numerical stability, we use the "max trick"
    m = torch.max(block_lse, lse)
    exp_old, exp_new = exp(lse - m), exp(block_lse - m)
    exp_sum = exp_old + exp_new
    
    new_lse = m + log(exp_sum)
    new_out = (out *exp_old +block_out * exp_new)/exp_sum

    return new_out, new_lse.squeeze(-1)

def update_out_and_lse_all_gather(
    out_dist: torch.Tensor,          # all float32
    lse_dist: torch.Tensor,
    use_log2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据每张卡 out, lse 计算最终结果 (Online Softmax 逻辑)
    out_dist: (B, H, S, CP, d)
    lse_dist: (B, H, S, CP)
    """
    cp_nranks = out_dist.shape[3]
    
    # 如果只有一个 rank/chunk，直接降维返回
    if cp_nranks == 1:
        return out_dist.squeeze(3), lse_dist.squeeze(3)

    # 选择对数和指数函数基底
    log, exp = (torch.log2, torch.exp2) if use_log2 else (torch.log, torch.exp)

    # 1. 为了数值稳定性，先找到 CP 维度上的最大 LSE (Max Trick)
    # lse_max shape: (B, H, S, 1)
    lse_max, _ = torch.max(lse_dist, dim=-1, keepdim=True)

    # 2. 计算每个 chunk 的权重因子 (unnormalized)
    # lse_exp shape: (B, H, S, CP)
    lse_exp = exp(lse_dist - lse_max)

    # 3. 计算分母（指数和）
    # lse_sum shape: (B, H, S, 1)
    lse_sum = lse_exp.sum(dim=-1, keepdim=True)

    # 4. 计算最终的 LSE
    # Final LSE = max + log(sum(exp(lse - max)))
    # shape: (B, H, S)
    final_lse = (lse_max + log(lse_sum)).squeeze(-1)

    # 5. 计算最终的 Output (加权平均)
    # Weights = exp(lse_i - max) / sum(exp(lse_j - max))
    # unsqueeze 为了广播到 hidden_dim (d) 维度: (B, H, S, CP, 1)
    weights = (lse_exp / lse_sum).unsqueeze(-1)

    # 执行加权求和
    # out_dist: (B, H, S, CP, d)
    # weights:  (B, H, S, CP, 1)
    # sum dim=3 -> (B, H, S, d)
    final_out = (out_dist * weights).sum(dim=3)

    return final_out, final_lse