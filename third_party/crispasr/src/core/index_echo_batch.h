#pragma once
#include "llama.h"
#include <cstdlib>

namespace core_index_echo {
// llama_batch_init allocates one position per token. Embedding batches for
// M-RoPE instead consume four position planes; audio uses the same sequential
// text position in all of them (it has no image/grid coordinates).
inline bool positions(llama_batch& batch, int first, int axes) {
    if (axes > 1) {
        free(batch.pos);
        batch.pos = (llama_pos*)malloc((size_t)batch.n_tokens * axes * sizeof(llama_pos));
        if (!batch.pos)
            return false;
    }
    for (int axis = 0; axis < axes; ++axis)
        for (int i = 0; i < batch.n_tokens; ++i)
            batch.pos[axis * batch.n_tokens + i] = first + i;
    return true;
}
} // namespace core_index_echo
