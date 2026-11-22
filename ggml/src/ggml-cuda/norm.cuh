#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx,
                       ggml_tensor *               dst,
                       ggml_tensor *               mul = nullptr,
                       ggml_tensor *               add = nullptr);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// q8_1_src1: the src1 of the quantized mat-vec that reads the result next, to also write its q8_1 copy (q8_1 cache)
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 const ggml_tensor * q8_1_src1 = nullptr);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     const ggml_tensor *         q8_1_src1 = nullptr,
                                     ggml_tensor *               scale_tensor = nullptr);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_should_fuse_l2_clip(const ggml_tensor * x, const ggml_tensor * scale, const ggml_tensor * dst);

void ggml_cuda_op_l2_clip_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * scale,
        const ggml_tensor * norm_scale, const ggml_tensor * fill, const ggml_tensor * clamp, ggml_tensor * dst);

#define GGML_CUDA_DEPTH_MIX_MAX_QUERY  4
#define GGML_CUDA_DEPTH_MIX_MAX_STATES 64

// stack [n_embd, n_stack, n_tokens] (or nullptr), cur [n_embd, n_tokens], query [n_embd, n_query], dst [n_embd, n_query, n_tokens]
void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * stack, const ggml_tensor * cur, const ggml_tensor * query,
        float eps, float scale, ggml_tensor * dst);
