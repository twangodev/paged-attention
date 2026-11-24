#pragma once

#include <vector>

// API for managing kv cache blocks in memory, a 'half' of the paged attention mechanism
// token t lives in slot s which is in block s/B in position s%B
class BlockManager {
public:
    BlockManager(int num_blocks, int block_size);

    int add_request();                    // returns block id
    int append_token(int req);            // returns slot
    void free_request(int req);

    int seq_len(int req) const;
    const std::vector<int>& block_table(int req) const;
    int num_free_blocks() const;

private:
    int block_size;
    std::vector<int> free_blocks;         // ids of unused blocks
    std::vector<std::vector<int>> tables; // tables[req] = block ids it owns
    std::vector<int> lens;                // lens[req] = tokens stored
};