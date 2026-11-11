#include "ggml.h"
#include "llama.h"

#include "common.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// the candidates of common_sampler_preselect must give the same top-k as all the logits
static void test_preselect(int n_vocab, int top_k, int last_n, float penalty_repeat, float penalty_freq, float penalty_present, int n_bias, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 3.0f);

    std::vector<float> logits(n_vocab);
    for (auto & l : logits) {
        l = dist(rng);
    }

    // the tokens with the largest logits, they are the worst case for the penalties and the biases
    std::vector<llama_token> top(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        top[i] = i;
    }
    const int n_top = std::min(n_vocab, 4*(top_k + last_n + n_bias) + 8);
    std::partial_sort(top.begin(), top.begin() + n_top, top.end(), [&](llama_token a, llama_token b) {
        return logits[a] > logits[b];
    });
    const auto pick = [&](int i) {
        return i % 2 == 0 ? top[rng() % n_top] : (llama_token) (rng() % n_vocab);
    };

    common_params_sampling params;
    params.samplers        = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_DRY, COMMON_SAMPLER_TYPE_TOP_N_SIGMA, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P };
    params.top_k           = top_k;
    params.penalty_last_n  = last_n;
    params.penalty_repeat  = penalty_repeat;
    params.penalty_freq    = penalty_freq;
    params.penalty_present = penalty_present;
    for (int i = 0; i < n_bias; i++) {
        const float bias = i % 3 == 0 ? -INFINITY : (i % 3 == 1 ? 4.0f : -2.5f);
        params.logit_bias.push_back({ pick(i), bias });
    }

    const size_t n = common_sampler_n_preselect(params, 0);
    GGML_ASSERT(n == (size_t) top_k + last_n + n_bias);

    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, params.logit_bias.size(), params.logit_bias.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, last_n, penalty_repeat, penalty_freq, penalty_present));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    for (int i = 0; i < last_n + 3; i++) {
        llama_sampler_accept(chain, pick(i));
    }

    std::vector<llama_token_data> cur_all(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur_all[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }
    llama_token_data_array cur_p_all = { cur_all.data(), cur_all.size(), -1, false };
    llama_sampler_apply(chain, &cur_p_all);

    std::vector<llama_token_data> cur_pre;
    common_sampler_preselect(logits.data(), n_vocab, n, params.logit_bias, cur_pre);
    GGML_ASSERT(cur_pre.size() >= n && cur_pre.size() <= n + n_bias);
    llama_token_data_array cur_p_pre = { cur_pre.data(), cur_pre.size(), -1, false };
    llama_sampler_apply(chain, &cur_p_pre);

    GGML_ASSERT(cur_p_all.size == (size_t) top_k);
    GGML_ASSERT(cur_p_pre.size == cur_p_all.size);
    for (size_t i = 0; i < cur_p_all.size; i++) {
        GGML_ASSERT(cur_p_pre.data[i].id    == cur_p_all.data[i].id);
        GGML_ASSERT(cur_p_pre.data[i].logit == cur_p_all.data[i].logit);
    }

    llama_sampler_free(chain);

    printf("[%s] n_vocab = %d, top_k = %d, last_n = %d, n_bias = %d: %zu of %d candidates\n", __func__, n_vocab, top_k, last_n, n_bias, cur_pre.size(), n_vocab);
}

static void test_preselect_disabled() {
    common_params_sampling params;
    params.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_DRY, COMMON_SAMPLER_TYPE_TOP_N_SIGMA, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P };
    params.top_k = 40;
    GGML_ASSERT(common_sampler_n_preselect(params, 3) == 43);

    {
        auto p = params;
        p.top_k = 0;
        GGML_ASSERT(common_sampler_n_preselect(p, 0) == 0);
    }
    {
        // a repeat penalty below 1 raises the logits of the tokens in the window
        auto p = params;
        p.penalty_last_n = 64;
        p.penalty_repeat = 0.9f;
        GGML_ASSERT(common_sampler_n_preselect(p, 0) == 0);
    }
    {
        auto p = params;
        p.penalty_last_n  = 64;
        p.penalty_present = -1.0f;
        GGML_ASSERT(common_sampler_n_preselect(p, 0) == 0);
    }
    {
        auto p = params;
        p.top_n_sigma = 1.0f;
        GGML_ASSERT(common_sampler_n_preselect(p, 0) == 0);
    }
    {
        auto p = params;
        p.dry_multiplier = 0.8f;
        GGML_ASSERT(common_sampler_n_preselect(p, 0) == 0);
    }
    {
        // top-p runs on all the logits when it comes first
        auto p = params;
        p.samplers = { COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TOP_K };
        GGML_ASSERT(common_sampler_n_preselect(p, 0) == 0);
    }
    {
        auto p = params;
        p.mirostat = 2;
        GGML_ASSERT(common_sampler_n_preselect(p, 0) == 0);
    }

    printf("[%s] OK\n", __func__);
}

int main(void) {
    test_preselect_disabled();
    for (uint32_t seed = 1; seed <= 20; seed++) {
        test_preselect(4096 + 61*seed, 20,  64, 1.1f, 0.2f, 1.5f, 7, seed);
        test_preselect(4096 + 61*seed,  1,   0, 1.0f, 0.0f, 0.0f, 0, seed);
        test_preselect(4096 + 61*seed, 40, 256, 1.0f, 0.0f, 1.5f, 0, seed);
        test_preselect(4096 + 61*seed,  5,  32, 1.3f, 0.0f, 0.0f, 9, seed);
    }
    test_preselect(250000, 20, 64, 1.0f, 0.0f, 1.5f, 3, 1234);

    return 0;
}
