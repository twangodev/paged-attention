#pragma once

// k_pool, v_pool: [num_blocks * block_size][num_heads][head_dim], one big GPU allocation.
// Slots come from BlockManager::append_token.

// Stores each request's newest k, v into its slot.
//   k_new, v_new: [batch_size][num_heads][head_dim]
//   slots:        [batch_size]
void write_kv_gpu(const float* k_new, const float* v_new, const int* slots, float* k_pool, float* v_pool, int batch_size, int num_heads, int head_dim);

// Same as attention_gpu, but retrieval of k/v values happens with a differen structure
//   q, out:       [batch_size][num_heads][head_dim]
//   block_tables: [batch_size][max_blocks_per_seq]
//   seq_lens:     [batch_size]
void paged_attention_gpu(const float* q, const float* k_pool, const float* v_pool, const int* block_tables, const int* seq_lens, float* out, int batch_size, int max_blocks_per_seq, int block_size, int num_heads, int head_dim);
