// The result of flash attention must not depend on the data of masked KV cells: a cache holds stale cells of
// removed tokens and the cells of other sequences. Each case runs twice on the device with different data in
// the cells that no token of the batch sees and compares the two results bit for bit.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct fattn_case {
    int64_t   hs;       // head size
    int64_t   nh_kv;    // KV heads
    int64_t   gqa;      // Q heads per KV head
    int64_t   n_tokens; // query tokens
    int64_t   n_kv;     // cells given to the op
    int64_t   n_valid;  // cells in front of the batch; the cells of the batch follow, the rest is masked
    bool      holes;    // every third cell in front of the batch is masked for all tokens
    bool      negative; // the masked cells hold only negative values in the second run
    ggml_type type_kv;
};

static bool run_case(ggml_backend_t backend, const fattn_case & c, bool verbose) {
    const size_t ctx_size = ggml_tensor_overhead()*16 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hs, c.n_tokens, c.nh_kv*c.gqa, 1);
    ggml_tensor * k0 = ggml_new_tensor_3d(ctx, c.type_kv, c.hs, c.nh_kv, c.n_kv);
    ggml_tensor * v0 = ggml_new_tensor_3d(ctx, c.type_kv, c.hs, c.nh_kv, c.n_kv);
    ggml_tensor * k  = ggml_permute(ctx, k0, 0, 2, 1, 3);
    ggml_tensor * v  = ggml_permute(ctx, v0, 0, 2, 1, 3);
    ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tokens, 1, 1);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) c.hs), 0.0f, 0.0f);

    if (!ggml_backend_supports_op(backend, out)) {
        if (verbose) {
            printf("hs=%3d gqa=%d n_tokens=%4d kv=%-5s: not supported\n", (int) c.hs, (int) c.gqa, (int) c.n_tokens, ggml_type_name(c.type_kv));
        }
        ggml_free(ctx);
        return true;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "failed to allocate the graph\n");
        return false;
    }

    std::mt19937 rng(1234 + (uint32_t) (c.hs*31 + c.n_tokens*7 + c.n_kv));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> q_data(ggml_nelements(q));
    for (auto & x : q_data) {
        x = dist(rng);
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));

    // token t sees the cells [0, n_valid + t] without the holes
    std::vector<ggml_fp16_t> m_data(ggml_nelements(m), ggml_fp32_to_fp16(-INFINITY));
    std::vector<bool> visible(c.n_kv, false); // to at least one token
    for (int64_t t = 0; t < c.n_tokens; t++) {
        for (int64_t i = 0; i <= c.n_valid + t && i < c.n_kv; i++) {
            if (c.holes && i < c.n_valid && i % 3 == 1) {
                continue;
            }
            m_data[t*c.n_kv + i] = ggml_fp32_to_fp16(0.0f);
            visible[i] = true;
        }
    }
    ggml_backend_tensor_set(m, m_data.data(), 0, ggml_nbytes(m));

    const size_t n_out = ggml_nelements(out);
    std::vector<float> res[2] = { std::vector<float>(n_out), std::vector<float>(n_out) };

    // the visible cells are the same in both runs
    std::vector<float> kv_f32[2] = { std::vector<float>(ggml_nelements(k0)), std::vector<float>(ggml_nelements(v0)) };
    for (auto & data : kv_f32) {
        for (auto & x : data) {
            x = dist(rng);
        }
    }

    for (int run = 0; run < 2; run++) {
        // the masked cells: zeros in the first run, other values in the second one
        for (auto & data : kv_f32) {
            for (int64_t i = 0; i < c.n_kv; i++) {
                if (visible[i]) {
                    continue;
                }
                for (int64_t j = 0; j < c.hs*c.nh_kv; j++) {
                    const float val = 3.0f*dist(rng);
                    data[i*c.hs*c.nh_kv + j] = run == 0 ? 0.0f : c.negative ? -fabsf(val) : val;
                }
            }
        }
        ggml_tensor * dst[2] = { k0, v0 };
        for (int t = 0; t < 2; t++) {
            if (c.type_kv == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(kv_f32[t].size());
                ggml_fp32_to_fp16_row(kv_f32[t].data(), tmp.data(), (int64_t) tmp.size());
                ggml_backend_tensor_set(dst[t], tmp.data(), 0, ggml_nbytes(dst[t]));
            } else {
                std::vector<uint8_t> tmp(ggml_nbytes(dst[t]));
                ggml_quantize_chunk(c.type_kv, kv_f32[t].data(), tmp.data(), 0, c.nh_kv*c.n_kv, c.hs, nullptr);
                ggml_backend_tensor_set(dst[t], tmp.data(), 0, tmp.size());
            }
        }

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return false;
        }
        ggml_backend_tensor_get(out, res[run].data(), 0, ggml_nbytes(out));
    }

    size_t n_diff   = 0;
    double max_diff = 0.0;
    bool   has_nan  = false;
    for (size_t i = 0; i < n_out; i++) {
        if (memcmp(&res[0][i], &res[1][i], sizeof(float)) != 0) {
            n_diff++;
            max_diff = std::max(max_diff, (double) fabsf(res[0][i] - res[1][i]));
        }
        has_nan = has_nan || std::isnan(res[0][i]) || std::isnan(res[1][i]);
    }

    const bool ok = n_diff == 0 && !has_nan;
    if (verbose || !ok) {
        printf("hs=%3d nh_kv=%d gqa=%d n_tokens=%4d n_kv=%6d n_valid=%6d holes=%d negative=%d kv=%-5s: %s (%zu of %zu values differ, max diff %.3g%s)\n",
            (int) c.hs, (int) c.nh_kv, (int) c.gqa, (int) c.n_tokens, (int) c.n_kv, (int) c.n_valid, c.holes, c.negative,
            ggml_type_name(c.type_kv), ok ? "OK" : "FAIL", n_diff, n_out, max_diff, has_nan ? ", NaN" : "");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return ok;
}

int main(int argc, char ** argv) {
    const bool verbose = argc > 1 && strcmp(argv[1], "-v") == 0;

    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_best();
    if (!backend) {
        fprintf(stderr, "no backend\n");
        return 1;
    }
    printf("backend: %s\n", ggml_backend_name(backend));

    int n_fail = 0;
    int n_run  = 0;
    for (ggml_type type_kv : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        for (int64_t hs : { 64, 128, 256, 512 }) {
            for (int64_t gqa : { 1, 4, 8 }) {
                for (int64_t n_tokens : { 1, 2, 3, 5, 8, 9, 20, 32, 512 }) {
                    for (int64_t n_valid : { 0, 100, 101, 1500 }) {
                        for (int variant = 0; variant < 4; variant++) {
                            const bool holes    = variant & 1;
                            const bool negative = variant & 2;
                            if (holes && n_valid == 0) {
                                continue;
                            }
                            // the caches hand out their cells in multiples of 256
                            const int64_t n_kv = GGML_PAD(n_valid + n_tokens + 300, 256);
                            const fattn_case c = { hs, 2, gqa, n_tokens, n_kv, n_valid, holes, negative, type_kv };
                            n_run++;
                            if (!run_case(backend, c, verbose)) {
                                n_fail++;
                            }
                        }
                    }
                }
            }
        }
    }

    printf("%d of %d cases failed\n", n_fail, n_run);

    ggml_backend_free(backend);

    return n_fail == 0 ? 0 : 1;
}
