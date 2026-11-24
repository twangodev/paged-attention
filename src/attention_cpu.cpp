#include "attention_cpu.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;


// Simple dot product between 2 flat arrays up to n
static float dot(const float* a, const float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void attention_cpu(const float* q, const float* k, const float* v, float* out, int seq_len, int num_heads, int head_dim) {
    float norm = 1 / pow(head_dim, 0.5);

    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            out[h * head_dim + d] = 0;
        }
    }

    vector<float> scores(seq_len);
    for (int h = 0; h < num_heads; h++) {
        // compute attention scores
        float m = -INFINITY;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = norm * dot(q + h*head_dim, k + (t*num_heads + h)*head_dim, head_dim);
            m = max(m, scores[t]);
        }

        // softmax
        float sum = 0;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = exp(scores[t] - m);
            sum += scores[t];
        }
        for (int t = 0; t < seq_len; t++) {
            scores[t] /= sum;
        }

        // update out[h]
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                out[h * head_dim + d] += scores[t] * v[(t*num_heads + h)*head_dim + d];
            }
        }
    }
}