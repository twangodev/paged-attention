#include "block_manager.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace std;

const int MAX_SEQ_LEN = 2048;
const int TOTAL_SLOTS = 64 * MAX_SEQ_LEN;  // kv memory: room for 64 max-length requests

// A fake user: tokens it arrives with, and tokens it will generate
struct Request {
    int prompt_len;
    int output_len;
};

struct Stats {
    int steps = 0;           // steps until every request finished
    double avg_running = 0;  // requests running at once, averaged over steps
    double waste = 0;        // fraction of reserved slots holding no token, averaged over steps
    int preemptions = 0;     // times a running request was kicked out to make room
};

// n requests with random lengths, prompt_len + output_len <= MAX_SEQ_LEN
vector<Request> make_requests(int n, mt19937& rng) {
    // TODO
    return {};
}

// Every step: admit waiting requests that fit, give each running request one more token
// (kick out the newest one if out of blocks), free the ones that are done.
Stats simulate(const vector<Request>& requests, int block_size) {
    BlockManager bm(TOTAL_SLOTS / block_size, block_size);
    Stats stats;
    // TODO
    return stats;
}

int main() {
    mt19937 rng(0);
    vector<Request> requests = make_requests(1000, rng);

    // block_size = MAX_SEQ_LEN is the reserve-max cache: every request gets one max-length block
    printf("block_size,steps,avg_running,waste,preemptions\n");
    for (int block_size : {8, 16, 32, 64, 128, MAX_SEQ_LEN}) {
        Stats s = simulate(requests, block_size);
        printf("%d,%d,%.1f,%.3f,%d\n", block_size, s.steps, s.avg_running, s.waste, s.preemptions);
    }
}
