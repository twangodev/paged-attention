#include "paged_attention.hpp"

// TODO: write_kv_kernel, one block per request, its threads copy num_heads * head_dim floats of k and v to the slot

void write_kv_gpu(const float* k_new, const float* v_new, const int* slots, float* k_pool, float* v_pool, int batch_size, int num_heads, int head_dim) {
    // TODO: launch write_kv_kernel
}

// TODO: paged_attention_kernel, copy of attention_kernel where only the k/v address of token t changes

void paged_attention_gpu(const float* q, const float* k_pool, const float* v_pool, const int* block_tables, const int* seq_lens, float* out, int batch_size, int max_blocks_per_seq, int block_size, int num_heads, int head_dim) {
    // TODO: launch paged_attention_kernel, same grid as attention_gpu, scores[] needs max_blocks_per_seq * block_size floats
}
