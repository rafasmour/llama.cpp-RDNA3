#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

// MUL of rows with one factor per row, read by the quantized mat-vec with the input q8_1_src1: one kernel for the product
// and the q8_1 copy of it (ggml_cuda_op_mul takes it when ctx.node_q8_1_src1 is set)
bool ggml_cuda_can_mul_row_factor_q8_1(const ggml_tensor * mul);
void ggml_cuda_op_mul_row_factor_q8_1(ggml_backend_cuda_context & ctx, ggml_tensor * mul, const ggml_tensor * q8_1_src1);
