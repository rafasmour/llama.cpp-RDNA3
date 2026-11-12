#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// CONCAT along dim 0 of two F32 tensors with few elements per row, fused with a CPY of the last rows of the result
// into a contiguous tensor (the conv state of a recurrent layer: the new input is appended and the state is stored)
bool ggml_cuda_can_concat_store(const ggml_tensor * concat, const ggml_tensor * view, const ggml_tensor * cpy);
void ggml_cuda_op_concat_store(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * cpy);
