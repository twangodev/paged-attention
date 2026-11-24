#pragma once

// Single-token attention for a batch of requests on the GPU, contiguous cache.
// All pointers are GPU memory. Flat row-major layouts:
//   q, out:           [batch_size][num_heads][head_dim]
//   k_cache, v_cache: [batch_size][max_seq_len][num_heads][head_dim]
//   seq_lens:         [batch_size]   tokens used by each request (<= max_seq_len)
void attention_gpu(const float* q, const float* k_cache, const float* v_cache, const int* seq_lens, float* out, int batch_size, int max_seq_len, int num_heads, int head_dim);