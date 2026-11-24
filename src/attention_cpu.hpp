#pragma once

// Single-token (decode) attention for one request. CPU reference.
// Flat row-major layouts:
//   q, out: [num_heads][head_dim]
//   k, v:   [seq_len][num_heads][head_dim]
void attention_cpu(const float* q, const float* k, const float* v, float* out, int seq_len, int num_heads, int head_dim);
