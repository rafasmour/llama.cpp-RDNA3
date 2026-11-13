#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// ssm_conv of a few tokens + the product with a gate of the same shape in one kernel, which can also write the q8_1 copy
// of the result for the mat-vec that reads it (q8_1_src1)
bool ggml_cuda_can_ssm_conv_mul(const ggml_tensor * conv, const ggml_tensor * mul);
void ggml_cuda_op_ssm_conv_mul(ggml_backend_cuda_context & ctx, ggml_tensor * conv, ggml_tensor * mul, const ggml_tensor * q8_1_src1);
