#pragma once

#include <cstdint>

template<bool Varlen=true>
struct BlockInfo {

    __device__ BlockInfo(int seqlen_q, int seqlen_k, const int bidb,
                         const int *cu_seqlens_q = nullptr,
                         const int *cu_seqlens_k = nullptr)
        : sum_s_q(!Varlen || cu_seqlens_q == nullptr ? -1 : cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || cu_seqlens_k == nullptr ? -1 : cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || cu_seqlens_q == nullptr ? seqlen_q : (cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb]))
        , actual_seqlen_k(!Varlen || cu_seqlens_k == nullptr ? seqlen_k : (cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb]))
    {}

    __forceinline__ __device__ int64_t q_offset(const int row_stride, const int bidb) const {
        return sum_s_q == -1 ? int64_t(bidb) * actual_seqlen_q * row_stride : int64_t(sum_s_q) * row_stride;
    }

    __forceinline__ __device__ int64_t k_offset(const int row_stride, const int bidb) const {
        return sum_s_k == -1 ? int64_t(bidb) * actual_seqlen_k * row_stride : int64_t(sum_s_k) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    const int actual_seqlen_k;
};

// Offset of (bidb, bidh, 0) in flash_attn's softmax LSE layout, also used for rowsum(dO * O):
// [batch, heads, seqlen_q], or [heads, total_q] when cu_seqlens_q packs the batch.
__forceinline__ __device__ int64_t lse_offset(const int *cu_seqlens_q, const int batch_size, const int num_heads,
                                              const int seqlen_q, const int bidb, const int bidh) {
    return cu_seqlens_q == nullptr
        ? (int64_t(bidb) * num_heads + bidh) * seqlen_q
        : int64_t(bidh) * cu_seqlens_q[batch_size] + cu_seqlens_q[bidb];
}
