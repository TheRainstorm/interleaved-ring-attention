    # tile_scheduler_metadata, num_splits = get_mla_metadata(
    #     cache_seqlens,
    #     1 * h_q // h_kv,
    #     h_kv,
    #     h_q,
    #     is_fp8=False,
    # )
    # out_ref2, lse = flash_mla_with_kvcache(
    #             q_local, kvcache_i, block_table, cache_seqlens, dv,
    #             tile_scheduler_metadata, num_splits,
    #             is_causal, is_fp8_kvcache, indices,
    #         )
        # # MLA相关元数据准备
    # cache_seqlens = torch.arange(1, S + 1, device=device, dtype=torch.int32)
    # cache_seqlens2 = torch.arange(0, S, device=device, dtype=torch.int32)
    