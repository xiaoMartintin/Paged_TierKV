#include <torch/extension.h>
#include <iostream>
#include <pybind11/stl.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

enum class BlockState {
    HOT = 0,
    WARM = 1,
    COLD = 2,
};

struct Block {
    int logical_id;
    int physical_idx;
    int seq_id;
    BlockState state;
};

class BlockManager {
public:
    explicit BlockManager(int total_blocks = 100000)
        : total_blocks_(total_blocks) {
        if (total_blocks_ <= 0) {
            throw std::runtime_error("total_blocks must be positive");
        }

        free_list_.reserve(total_blocks_);
        for (int idx = total_blocks_ - 1; idx >= 0; --idx) {
            free_list_.push_back(idx);
        }

        std::cout << "[Cloud C++] Block Manager Initialized on Modal GPU with "
                  << total_blocks_ << " physical blocks." << std::endl;
    }

    int allocate_block(int seq_id) {
        if (free_list_.empty()) {
            throw std::runtime_error("No free physical blocks available.");
        }

        int physical_idx = free_list_.back();
        free_list_.pop_back();

        auto& sequence_blocks = block_table_[seq_id];
        int logical_id = static_cast<int>(sequence_blocks.size());
        sequence_blocks.push_back(Block{logical_id, physical_idx, seq_id, BlockState::HOT});

        std::cout << "[Cloud C++] allocate_block: seq_id=" << seq_id
                  << ", logical_id=" << logical_id
                  << ", physical_idx=" << physical_idx
                  << ", state=HOT" << std::endl;

        return physical_idx;
    }

    void free_sequence(int seq_id) {
        auto table_it = block_table_.find(seq_id);
        if (table_it == block_table_.end()) {
            std::cout << "[Cloud C++] free_sequence: seq_id=" << seq_id
                      << " not found, nothing to free." << std::endl;
            return;
        }

        for (const auto& block : table_it->second) {
            free_list_.push_back(block.physical_idx);
            std::cout << "[Cloud C++] free_sequence: returned physical_idx="
                      << block.physical_idx << " for seq_id=" << seq_id << std::endl;
        }

        block_table_.erase(table_it);
        std::cout << "[Cloud C++] free_sequence: cleared seq_id=" << seq_id << std::endl;
    }

    void update_block_state(int seq_id, int logical_id, int new_state_int) {
        auto table_it = block_table_.find(seq_id);
        if (table_it == block_table_.end()) {
            throw std::runtime_error("Sequence not found for state update.");
        }

        auto& blocks = table_it->second;
        if (logical_id < 0 || logical_id >= static_cast<int>(blocks.size())) {
            throw std::runtime_error("Logical block ID out of range.");
        }

        BlockState new_state = parse_state(new_state_int);
        blocks[logical_id].state = new_state;

        std::cout << "[Cloud C++] update_block_state: seq_id=" << seq_id
                  << ", logical_id=" << logical_id
                  << ", new_state=" << new_state_int << std::endl;
    }

    std::vector<int> get_physical_indices(int seq_id) const {
        auto table_it = block_table_.find(seq_id);
        if (table_it == block_table_.end()) {
            return {};
        }

        std::vector<int> physical_indices;
        physical_indices.reserve(table_it->second.size());
        for (const auto& block : table_it->second) {
            physical_indices.push_back(block.physical_idx);
        }
        return physical_indices;
    }

    std::vector<int> get_states(int seq_id) const {
        auto table_it = block_table_.find(seq_id);
        if (table_it == block_table_.end()) {
            return {};
        }

        std::vector<int> states;
        states.reserve(table_it->second.size());
        for (const auto& block : table_it->second) {
            states.push_back(static_cast<int>(block.state));
        }
        return states;
    }

private:
    static BlockState parse_state(int new_state_int) {
        switch (new_state_int) {
            case 0:
                return BlockState::HOT;
            case 1:
                return BlockState::WARM;
            case 2:
                return BlockState::COLD;
            default:
                throw std::runtime_error("Invalid BlockState value.");
        }
    }

    int total_blocks_;
    std::vector<int> free_list_;
    std::unordered_map<int, std::vector<Block>> block_table_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BlockManager>(m, "BlockManager")
        .def(py::init<int>(), py::arg("total_blocks") = 100000)
        .def("allocate_block", &BlockManager::allocate_block)
        .def("free_sequence", &BlockManager::free_sequence)
        .def("update_block_state", &BlockManager::update_block_state)
        .def("get_physical_indices", &BlockManager::get_physical_indices)
        .def("get_states", &BlockManager::get_states);
}
