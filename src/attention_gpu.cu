#include "attention_gpu.hpp"

#include <cmath>

// same as cpu dot
__device__ float dot(const float* a, const float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// One block per (head, request). For now each block has one thread, which does everything
__global__ void attention_kernel(const float* q, const float* k_cache, const float* v_cache, const int* seq_lens, float* out, int max_seq_len, int num_heads, int head_dim) {
    int h = blockIdx.x;  // head
    int r = blockIdx.y;  // request
    int len = seq_lens[r];

    size_t chunk = (size_t)r * max_seq_len * num_heads * head_dim;
    const float* k = k_cache + chunk;
    const float* v = v_cache + chunk;
    const float* q_row = q + (r * num_heads + h) * head_dim;
    float* out_row = out + (r * num_heads + h) * head_dim;

    extern __shared__ float scores[];
    float norm = 1.0f / sqrtf(head_dim);

    // attention scores calc
    int m = -INFINITY;
    for (int t = 0; t < len; t++) {
        scores[t] = norm * dot(q_row, k + (t*num_heads + h)*head_dim, head_dim);
        m = fmaxf(m, scores[t]);
    }

    // softmax
    float sum = 0;
    for (int t = 0; t < len; t++) {
        scores[t] = expf(scores[t] - m);
        sum += scores[t];
    }
    for (int t = 0; t < len; t++) {
        scores[t] /= sum;
    }

    // blend the attention
    for (int d = 0; d < head_dim; d++) {
        float acc = 0;
        for (int t = 0; t < len; t++) {
            acc += scores[t] * v[(t*num_heads + h)*head_dim + d];
        }
        out_row[d] = acc;
    }
}

// Runs on the CPU: picks the grid shape and launches the kernel.
void attention_gpu(const float* q, const float* k_cache, const float* v_cache, const int* seq_lens, float* out,
                   int num_seqs, int max_seq_len, int num_heads, int head_dim) {
    dim3 grid(num_heads, num_seqs);
    int threads = 1;
    size_t shared_bytes = max_seq_len * sizeof(float);   // size of scores[]
    attention_kernel<<<grid, threads, shared_bytes>>>(q, k_cache, v_cache, seq_lens, out, max_seq_len, num_heads, head_dim);
}