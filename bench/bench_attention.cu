#include "attention_gpu.hpp"
#include "paged_attention.hpp"

#include <cstdio>

const int H = 32, D = 128, REPS = 10;

// Average ms per call of run(), after one warmup call
template <class F>
float time_ms(F run) {
    // TODO: time REPS calls with cudaEventRecord / cudaEventElapsedTime
    return 0;
}

// k, v: [batch_size][seq_len][H][D]
float time_contiguous(int batch_size, int seq_len) {
    // TODO: allocate q, k, v, out, seq_lens on the gpu, time attention_gpu, free
    return 0;
}

// k_pool, v_pool: [batch_size * seq_len][H][D], each request's blocks scattered in random order
float time_paged(int batch_size, int seq_len, int block_size) {
    // TODO: same as time_contiguous plus shuffled block_tables, time paged_attention_gpu
    return 0;
}

// gb_per_s = bytes of k and v read / time
void print_row(const char* kernel, int batch_size, int seq_len, int block_size, float ms) {
    // TODO
}

int main() {
    printf("kernel,batch_size,seq_len,block_size,ms,gb_per_s\n");
    for (int batch_size : {8, 32}) {
        for (int seq_len : {512, 2048}) {
            print_row("contiguous", batch_size, seq_len, seq_len, time_contiguous(batch_size, seq_len));
            for (int block_size : {8, 16, 32, 64, 128}) {
                print_row("paged", batch_size, seq_len, block_size, time_paged(batch_size, seq_len, block_size));
            }
        }
    }
}
