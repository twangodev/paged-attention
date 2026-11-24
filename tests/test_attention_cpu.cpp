#include "attention_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace std;

static int failures = 0;

// Prints ok/FAIL. Fails if any got[i] is more than 1e-5 away from want[i] (NaN fails too).
static void expect_close(const char* name, const float* got, const float* want, int n) {
    for (int i = 0; i < n; i++) {
        if (!(fabs(got[i] - want[i]) <= 1e-5f)) {
            printf("FAIL %s: index %d got %f want %f\n", name, i, got[i], want[i]);
            failures++;
            return;
        }
    }
    printf("ok   %s\n", name);
}

// Small enough to do on paper. head_dim = 4, so scale = 0.5.
// scores = 0.5 * [2, 0, 1] = [1, 0, 0.5] -> softmax -> [0.5065, 0.1863, 0.3072].
// Each v is one-hot, so out is just those three weights.
static void test_hand_example() {
    float q[] = {1, 1, 0, 0};
    float k[] = {1, 1, 0, 0,
                 0, 0, 1, 1,
                 1, 0, 0, 0};
    float v[] = {1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0};
    float want[] = {0.506480f, 0.186324f, 0.307196f, 0.0f};
    float out[4];
    attention_cpu(q, k, v, out, 3, 1, 4);
    expect_close("hand example", out, want, 4);
}

// Token 7's key is 100 * q, so its score is ~250 and it takes ~all the weight: out = v[7].
// exp(250) overflows a float, so this only passes if the max is subtracted before exp.
static void test_big_score() {
    const int S = 20, H = 2, D = 64, winner = 7;
    mt19937 rng(0);
    uniform_real_distribution<float> dist(-1, 1);
    vector<float> q(H * D), k(S * H * D), v(S * H * D), out(H * D);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);
    for (int i = 0; i < H * D; i++) k[winner * H * D + i] = 100 * q[i];

    attention_cpu(q.data(), k.data(), v.data(), out.data(), S, H, D);
    expect_close("big score", out.data(), &v[winner * H * D], H * D);
}

int main() {
    test_hand_example();
    test_big_score();
    return failures == 0 ? 0 : 1;
}