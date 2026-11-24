// Checks attention_gpu against attention_cpu on a batch of requests with different lengths.
#include "attention_cpu.hpp"
#include "attention_gpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace std;

// Stops the program if a CUDA call failed. Without this, CUDA errors are silent.
static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// Makes room on the GPU, copies a CPU vector into it, and returns the GPU pointer.
// (template = works for vector<float> and vector<int> alike)
template <typename T>
static T* to_gpu(const vector<T>& x) {
    T* dev;
    check(cudaMalloc(&dev, x.size() * sizeof(T)));
    check(cudaMemcpy(dev, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

int main() {
    const int H = 4, D = 64, max_len = 300;
    vector<int> lens = {1, 300, 129, 77, 256};  // includes 1, max_len, and just over 128
    const int S = lens.size();
    const int chunk = max_len * H * D;          // floats per request in the cache

    // Random q and cache. Unused cache slots (t >= lens[s]) are NaN, so if the
    // kernel ever reads one, its answer turns into NaN and the test fails.
    mt19937 rng(0);
    uniform_real_distribution<float> dist(-1, 1);
    vector<float> q(S * H * D), k(S * chunk, NAN), v(S * chunk, NAN);
    for (float& x : q) x = dist(rng);
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < lens[s] * H * D; i++) {
            k[s * chunk + i] = dist(rng);
            v[s * chunk + i] = dist(rng);
        }
    }

    // GPU: copy in, run, copy out.
    float* q_d = to_gpu(q);
    float* k_d = to_gpu(k);
    float* v_d = to_gpu(v);
    int* lens_d = to_gpu(lens);
    float* out_d;
    check(cudaMalloc(&out_d, S * H * D * sizeof(float)));

    attention_gpu(q_d, k_d, v_d, lens_d, out_d, S, max_len, H, D);
    check(cudaGetLastError());       // did the launch work?
    check(cudaDeviceSynchronize());  // wait for the kernel to finish; did it crash?

    vector<float> got(S * H * D);
    check(cudaMemcpy(got.data(), out_d, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU answer: request s's piece of each array looks exactly like step 1's inputs.
    vector<float> want(S * H * D);
    for (int s = 0; s < S; s++) {
        attention_cpu(&q[s * H * D], &k[s * chunk], &v[s * chunk], &want[s * H * D], lens[s], H, D);
    }

    // Compare, and say where the first mismatch is.
    int bad = 0;
    for (int i = 0; i < S * H * D; i++) {
        if (!(fabs(got[i] - want[i]) <= 1e-5f)) {
            if (bad == 0) {
                printf("FAIL first mismatch at request %d, head %d, d %d: got %f want %f\n",
                       i / (H * D), i / D % H, i % D, got[i], want[i]);
            }
            bad++;
        }
    }
    if (bad == 0) printf("ok   gpu matches cpu\n");
    else printf("FAIL %d of %d numbers wrong\n", bad, S * H * D);

    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
    cudaFree(lens_d);
    cudaFree(out_d);
    return bad == 0 ? 0 : 1;
}