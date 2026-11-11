#pragma once

#include "ggml-backend.h"

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

// Writes a batch of tensor ranges from host memory, returns false if the caller has to copy them one by one.
bool ggml_cuda_set_tensor_n(int device,
        struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

