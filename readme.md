## Description

Interleaved Ring Attention optimizes standard Ring Attention by fundamentally changing how tokens are partitioned across GPUs. Instead of splitting sequences into contiguous chunks, this algorithm distributes Query ($q$), Key ($k$), and Value ($v$) tokens to gpu using a modulo strategy:

$$

\texttt{GPU}_\texttt{id} = \texttt{token}_\texttt{idx} \pmod {\texttt{cp\_nrank}}

$$

Key Features:

- Perfect Load Balancing: The round-robin distribution ensures every GPU processes a uniform number of tokens, eliminating computational skew caused by uneven sequence lengths or causal masking.
- Simplicity: Minimal code changes required compared to [DualChunkSwap](https://github.com/NVIDIA/TransformerEngine/blob/v2.5/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L3782) strategy

## Principle

Assume 4 gpus and a sequence length of 16, each gpu will receive tokens q,k,v with ids: [$q_i, q_i + 4, q_i + 8, q_i + 12$], where $q_i$ is gpu cp rank id. 

For prefill ring-attention, each gpu will exchange k,v tokens with other gpus in each step. And the q, kv mask is acctully nearly a standard causal mask. The mask between q tokens (cp rank id=$p_i$) and kv tokens (cp rank id=$p_j$) is show as below:

![image.png](https://raw.githubusercontent.com/TheRainstorm/.image-bed/main/20251123194059.png)

When $p_i \ge p_j$, it's a standard casual attention. When $p_i < p_j$, the attention can be computed by skip the q's first token and the kv's last token. Since all mask is nearly standard casual mask, the computation load on each gpu is nearly equal.

## Python proof-of-code

See [interleaved_ring_attention.py], which verifies the interleaved ring attention implementation and compares its output with standard attention.
