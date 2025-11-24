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
    
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(o1.flatten(), o2.flatten(), dim=0).item()
    
    print(f'{msg:35s}: CosSim={cos_sim:.6f}, MAE={mean_abs_error:.6f}, MaxError={max_error:.6f}')
    
def attention_causal_with_lse(query_states, key_states, value_states,
                              scale=None, causal=True, use_log2=False):
    """
    计算单块内的因果注意力，返回 Output, 标准 LSE, 和 Log2 LSE
    Input Shape: (B, H, S, d)
    """
    B, H, S, d = query_states.shape
    device = query_states.device
    dtype = query_states.dtype

    if scale is None:
        scale = d ** -0.5
    
    attn_scores = query_states @ key_states.transpose(-2, -1)  # (B, H, S, S)
    attn_scores = attn_scores.float()
    attn_scores = attn_scores * scale

    if causal:
        mask = torch.triu(torch.ones(S, S, device=query_states.device, dtype=torch.bool), diagonal=1)
        attn_scores.masked_fill_(mask, float('-inf'))

    if not use_log2:
        # 1. 标准 LSE (base e)
        lse = torch.logsumexp(attn_scores, dim=-1)  # (B, H, S)
        attn_weights = attn_scores.softmax(dim=-1)
    else:
        # 2. Base-2 LSE
        # log2(sum(2^x)) = logsumexp(x * ln(2)) / ln(2)
        ln_2 = math.log(2.0)
        lse = torch.logsumexp(attn_scores * ln_2, dim=-1) / ln_2
        attn_weights = (attn_scores * ln_2).softmax(dim=-1)  # log2 时, out 也要使用基于 2 的 softmax 加权
    
    attn_weights = attn_weights.to(dtype)
    out = attn_weights @ value_states  # (B, H, S, d)

    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    use_log2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if out is None:
        return block_out.clone(), block_lse.clone()
    
    return _update_out_and_lse_impl(out, lse, block_out, block_lse, use_log2)

def _update_out_and_lse_impl(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    use_log2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在线更新 Ring Attention 的累积结果 (Online Softmax 逻辑)
    """
    dtype = out.dtype
    out, block_out = out.float(), block_out.float()

    block_lse = block_lse.unsqueeze(dim=-1) #(B, H, S, 1)
    lse = lse.unsqueeze(dim=-1)             #(B, H, S, 1)

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
    
    # ref: https://github.com/zhuzilin/ring-flash-attention
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    # new_out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    # new_lse = lse - F.logsigmoid(lse - block_lse)

    new_out = new_out.to(dtype)
    return new_out, new_lse.squeeze(-1)
