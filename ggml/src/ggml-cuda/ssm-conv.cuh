#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// ssm_conv of a few tokens + the product with a gate of the same shape in one kernel, which can also write the q8_1 copy
// of the result for the mat-vec that reads it (q8_1_src1)
bool ggml_cuda_can_ssm_conv_mul(const ggml_tensor * conv, const ggml_tensor * mul);
void ggml_cuda_op_ssm_conv_mul(ggml_backend_cuda_context & ctx, ggml_tensor * conv, ggml_tensor * mul, const ggml_tensor * q8_1_src1);

// taps of ggml_cuda_op_dyn_conv_tail: one strided [1, n_head, n_seq_tokens, n_seqs] view and one row offset of x per lag
#define GGML_CUDA_DYN_CONV_MAX_TAPS 8
struct ggml_cuda_dyn_conv_taps {
    const float * data[GGML_CUDA_DYN_CONV_MAX_TAPS];
    int64_t nb1[GGML_CUDA_DYN_CONV_MAX_TAPS]; // in floats
    int64_t nb2[GGML_CUDA_DYN_CONV_MAX_TAPS];
    int64_t nb3[GGML_CUDA_DYN_CONV_MAX_TAPS];
    int     row[GGML_CUDA_DYN_CONV_MAX_TAPS];
    int     n;
    int     n_conv;
};

void ggml_cuda_op_dyn_conv_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * base, const ggml_tensor * cur,
        const ggml_cuda_dyn_conv_taps & taps, int head_dim, const ggml_tensor * scale_node, ggml_tensor * dst);
