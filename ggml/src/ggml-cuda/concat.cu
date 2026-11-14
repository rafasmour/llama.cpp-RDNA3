#include "concat.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// non-contiguous kernel for short rows: the threads of a block are spread over (i0, i1), a block per row would be mostly idle
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont_short(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t   ne1,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;

    const uint32_t i  = blockIdx.x*blockDim.x + threadIdx.x;
    const uint32_t i1 = i / (uint32_t) ne0;

    if (i1 >= ne1) {
        return;
    }

    const T * x;

    {
        const int64_t i0 = i - i1*(uint32_t) ne0;

        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        // rows much shorter than a block, e.g. the conv state of a few tokens: one block per row would leave most threads idle
        const bool short_rows = dst->ne[0] <= CUDA_CONCAT_BLOCK_SIZE/4 && dst->ne[0]*dst->ne[1] <= INT32_MAX;

        const dim3 grid_dim(short_rows ? (dst->ne[0]*dst->ne[1] + CUDA_CONCAT_BLOCK_SIZE - 1)/CUDA_CONCAT_BLOCK_SIZE : dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            if (short_rows) {
                concat_non_cont_short<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
                return;
            }
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
        }
    }
}

#define CUDA_CONCAT_STORE_MAX_HEAD 16

// dst[row] = { src0[row][0..n0), src1[row][0..n1) }, store[row] = the last n0 elements of dst[row].
// One thread per row; store may be the memory that src0 is read from, so a thread reads its src0 row before it writes.
// a: the src0 row, src1_at(t): element t of the src1 row, d: the dst row, st: the store row
template <typename F>
static __device__ __forceinline__ void concat_dim0_store_row(const float * a, const F & src1_at, float * d, float * st, const int n0, const int n1) {
    float head[CUDA_CONCAT_STORE_MAX_HEAD];
#pragma unroll
    for (int k = 0; k < CUDA_CONCAT_STORE_MAX_HEAD; ++k) {
        head[k] = k < n0 ? a[k] : 0.0f;
    }

#pragma unroll
    for (int k = 0; k < CUDA_CONCAT_STORE_MAX_HEAD; ++k) {
        if (k < n0) {
            d[k] = head[k];
        }
    }
    for (int t = 0; t < n1; ++t) {
        d[n0 + t] = src1_at(t);
    }
#pragma unroll
    for (int k = 0; k < CUDA_CONCAT_STORE_MAX_HEAD; ++k) {
        if (k < n0) {
            const int j = n1 + k; // column of the result
            st[k] = j < n0 ? head[j] : src1_at(j - n0);
        }
    }
}

static __global__ void concat_dim0_store_f32(
        const float * src0, const float * src1, float * dst, float * store,
        const int n0, const int n1, const int nrows,
        const int64_t s0_2, const int64_t s1_0, const int64_t s1_1, const int64_t s1_2,
        const int64_t dst_2, const int64_t store_2) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;
    if (row >= nrows) {
        return;
    }

    const float * b = src1 + seq * s1_2 + (int64_t) row * s1_1;

    concat_dim0_store_row(src0 + seq * s0_2 + (int64_t) row * n0, [&](const int t) { return b[t * s1_0]; },
        dst + seq * dst_2 + (int64_t) row * (n0 + n1), store + seq * store_2 + (int64_t) row * n0, n0, n1);
}

// the same with src1 = the product of two tensors with the same strides (B * x of the LFM2 short conv)
static __global__ void concat_dim0_mul_store_f32(
        const float * src0, const float * src1, const float * src1_mul, float * dst, float * store,
        const int n0, const int n1, const int nrows,
        const int64_t s0_2, const int64_t s1_0, const int64_t s1_1, const int64_t s1_2,
        const int64_t dst_2, const int64_t store_2) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;
    if (row >= nrows) {
        return;
    }

    const float * b = src1     + seq * s1_2 + (int64_t) row * s1_1;
    const float * m = src1_mul + seq * s1_2 + (int64_t) row * s1_1;

    concat_dim0_store_row(src0 + seq * s0_2 + (int64_t) row * n0, [&](const int t) { return b[t * s1_0] * m[t * s1_0]; },
        dst + seq * dst_2 + (int64_t) row * (n0 + n1), store + seq * store_2 + (int64_t) row * n0, n0, n1);
}

bool ggml_cuda_can_concat_store(const ggml_tensor * concat, const ggml_tensor * view, const ggml_tensor * cpy) {
    const ggml_tensor * src0 = concat->src[0];
    const ggml_tensor * src1 = concat->src[1];

    if (((const int32_t *) concat->op_params)[0] != 0 ||
            concat->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || cpy->type != GGML_TYPE_F32 ||
            concat->ne[3] != 1 || !ggml_is_contiguous(concat) || !ggml_is_contiguous(src0)) {
        return false;
    }

    const int64_t n0 = src0->ne[0];
    const int64_t n1 = src1->ne[0];
    // the rows are short: a state of a few columns and the tokens of a small batch
    if (n0 < 1 || n0 > CUDA_CONCAT_STORE_MAX_HEAD || n1 < 1 || n1 > 8) {
        return false;
    }

    // the copy source is the last n0 columns of every row of the result
    if (view->view_src != concat || view->view_offs != (size_t) n1 * sizeof(float) ||
            view->ne[0] != n0 || view->ne[1] != concat->ne[1] || view->ne[2] != concat->ne[2] || view->ne[3] != 1 ||
            view->nb[0] != concat->nb[0] || view->nb[1] != concat->nb[1] || view->nb[2] != concat->nb[2]) {
        return false;
    }
    if (cpy->src[0] != view) {
        return false;
    }

    // the copy destination holds the rows of a sequence one after another
    if (cpy->nb[0] != sizeof(float) || cpy->ne[0] != n0 * concat->ne[1] || cpy->ne[1] != concat->ne[2] ||
            cpy->ne[2] != 1 || cpy->ne[3] != 1 || cpy->nb[1] % sizeof(float) != 0) {
        return false;
    }

    return src1->nb[0] % sizeof(float) == 0 && src1->nb[1] % sizeof(float) == 0 && src1->nb[2] % sizeof(float) == 0;
}

bool ggml_cuda_can_concat_store_mul(const ggml_tensor * concat, const ggml_tensor * mul) {
    const ggml_tensor * src1 = concat->src[1];
    const ggml_tensor * a    = mul->src[0];
    const ggml_tensor * b    = mul->src[1];

    // src1 is the transposed product, the factors have the shape and the strides of each other
    if (src1->op != GGML_OP_TRANSPOSE || src1->src[0] != mul || src1->data != mul->data ||
            mul->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || mul->ne[3] != 1 ||
            !ggml_are_same_shape(mul, a) || !ggml_are_same_shape(mul, b)) {
        return false;
    }
    return a->nb[0] == sizeof(float) && a->nb[1] % sizeof(float) == 0 && a->nb[2] % sizeof(float) == 0 &&
           a->nb[0] == b->nb[0] && a->nb[1] == b->nb[1] && a->nb[2] == b->nb[2];
}

void ggml_cuda_op_concat_store(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * cpy, const ggml_tensor * mul) {
    const ggml_tensor * src0 = concat->src[0];
    const ggml_tensor * src1 = concat->src[1];

    const int n0    = src0->ne[0];
    const int n1    = src1->ne[0];
    const int nrows = concat->ne[1];
    const int nseqs = concat->ne[2];

    const dim3 block_nums((nrows + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE, nseqs, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, CUDA_CONCAT_BLOCK_SIZE, 0, ctx.stream());
    if (mul != nullptr) {
        // the factors are read where the transposed product would be: dims 0 and 1 swapped
        const ggml_tensor * a = mul->src[0];
        const ggml_tensor * b = mul->src[1];
        ggml_cuda_kernel_launch(concat_dim0_mul_store_f32, launch_params,
            (const float *) src0->data, (const float *) a->data, (const float *) b->data, (float *) concat->data, (float *) cpy->data,
            n0, n1, nrows,
            (int64_t) (src0->nb[2] / sizeof(float)),
            (int64_t) (a->nb[1] / sizeof(float)), (int64_t) (a->nb[0] / sizeof(float)), (int64_t) (a->nb[2] / sizeof(float)),
            (int64_t) (concat->nb[2] / sizeof(float)), (int64_t) (cpy->nb[1] / sizeof(float)));
        return;
    }
    ggml_cuda_kernel_launch(concat_dim0_store_f32, launch_params,
        (const float *) src0->data, (const float *) src1->data, (float *) concat->data, (float *) cpy->data,
        n0, n1, nrows,
        (int64_t) (src0->nb[2] / sizeof(float)),
        (int64_t) (src1->nb[0] / sizeof(float)), (int64_t) (src1->nb[1] / sizeof(float)), (int64_t) (src1->nb[2] / sizeof(float)),
        (int64_t) (concat->nb[2] / sizeof(float)), (int64_t) (cpy->nb[1] / sizeof(float)));
}
