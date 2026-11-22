#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// res [n0, n_used, n_tokens] F32, scale [n_expert] F32, ids [n_used, n_tokens] I32
void ggml_cuda_op_moe_expert_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * res, const ggml_tensor * scale, const ggml_tensor * ids,
        ggml_tensor * dst);
