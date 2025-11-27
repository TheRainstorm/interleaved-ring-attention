## Description

Interleaved Ring Attention optimizes standard Ring Attention by fundamentally changing how tokens are partitioned across GPUs. Instead of splitting sequences into contiguous chunks, this algorithm distributes Query ($q$), Key ($k$), and Value ($v$) tokens to gpu using a modulo strategy:

$$
\texttt{GPU}_\texttt{id} = \texttt{token}_\texttt{idx}  \pmod {\texttt{CP}}
$$

Key Features:

- **Perfect Load Balancing**: The round-robin distribution ensures every GPU processes a uniform number of tokens, eliminating computational skew caused by uneven sequence lengths or causal masking.
- **Simplicity**: Minimal code changes required compared to [DualChunkSwap](https://github.com/NVIDIA/TransformerEngine/blob/v2.5/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L3782) strategy

## Principle

Assume 4 gpus and a sequence length of 16, each gpu will receive tokens q,k,v with ids: [ $q_i, q_i + 4, q_i + 8, q_i + 12$ ], where $q_i$ is gpu cp rank id. 

For prefill ring-attention, each gpu will exchange k,v tokens with other gpus in each step. And the q, kv mask is acctully nearly a standard causal mask. The mask between q tokens (cp rank id= $p_i$ ) and kv tokens (cp rank id= $p_j$ ) is show as below:

![image.png](https://raw.githubusercontent.com/TheRainstorm/.image-bed/main/20251123194059.png)

When $p_i \ge p_j$, it's a standard casual attention. When $p_i < p_j$, the attention can be computed by skip the q's first token and the kv's last token. Since all mask is nearly standard casual mask, the computation load on each gpu is nearly equal.

## Python proof-of-code

See interleaved_ring_attention.py, which verifies the interleaved ring attention implementation and compares its output with standard attention.

### about log2 lse

In Ring Attention, the cumulative out and lse (log-sum-exp) values are updated with the out and lse from the current block. The standard update formulas are base-e:

$$
out_{new} = \frac{out_1 \exp(lse_1) + out_2 \exp(lse_2)}{\exp(lse_1)+\exp(lse_2)}\\
lse_{new} = \log{(\exp(lse_i)+\exp(lse_i))}
$$

While implementing Ring Attention using flashinfer as the attention library, I encountered incorrect results. After inspecting the `merge_state` kernel in the flashinfer source code, I noticed that it replaces all exp and log operations with their base-2 equivalents (exp2, log2).

My initial hypothesis was that the formulas would remain correct as long as all exp and log operations within the LSE calculation were consistently base-2. To test this, I created a Python simulation where my PyTorch-based attention function returned `lse(x) = log2(sum(2^x))`. However, the Ring Attention verification using this LSE failed. Curiously, the verification succeeded if the out value from the attention function (`out = softmax(QK)V`) was also computed using a base-2 softmax. This behavior seemed strange.

The puzzle was solved when I examined the LSE implementation in FlashMLA. I realized that the correct LSE definition in this context is `lse(x) = log2(sum(e^x))`. When using this definition, the out value can be computed with the standard, base-e softmax, and the final results match perfectly.