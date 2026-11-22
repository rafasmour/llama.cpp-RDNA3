#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

// per-expert weight scale of a MUL_MAT_ID result: dst[:, k, t] = res[:, k, t] * scale[ids[k, t]] (REPEAT, GET_ROWS and MUL)
static __global__ void moe_expert_scale_f32(const float * __restrict__ res, const float * __restrict__ scale, const int32_t * __restrict__ ids,
        float * __restrict__ dst, const int64_t n0, const int n_used, const int n_expert, const int64_t ids_nb0, const int64_t ids_nb1, const int64_t n) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t row = i / n0;
    const int     k   = row % n_used;
    const int64_t t   = row / n_used;
    const int32_t id  = ids[k*ids_nb0 + t*ids_nb1];
    dst[i] = res[i] * (uint32_t(id) < uint32_t(n_expert) ? scale[id] : 0.0f);
}

void ggml_cuda_op_moe_expert_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * res, const ggml_tensor * scale, const ggml_tensor * ids,
        ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst);
    const int block_size = 256;
    const int64_t n_blocks = (n + block_size - 1) / block_size;
    GGML_ASSERT(n_blocks <= INT_MAX);
    moe_expert_scale_f32<<<n_blocks, block_size, 0, ctx.stream()>>>((const float *) res->data, (const float *) scale->data, (const int32_t *) ids->data,
        (float *) dst->data, res->ne[0], res->ne[1], ggml_nelements(scale), ids->nb[0]/sizeof(int32_t), ids->nb[1]/sizeof(int32_t), n);
}
