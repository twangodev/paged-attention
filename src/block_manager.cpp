#include "block_manager.hpp"

BlockManager::BlockManager(int num_blocks, int block_size) : block_size(block_size) {
    // TODO
}

int BlockManager::add_request() {
    // TODO return block id
    return -1;
}

int BlockManager::append_token(int req) {
    // TODO: return the slot
    return -1;
}

void BlockManager::free_request(int req) {
    // TODO: free all table[req]
}

int BlockManager::seq_len(int req) const {
    return lens[req];
}

const std::vector<int>& BlockManager::block_table(int req) const {
    return tables[req];
}

int BlockManager::num_free_blocks() const {
    return free_blocks.size();
}
