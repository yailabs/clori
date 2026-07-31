
#include <string.h>

#include <yvex/gguf.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/gguf.h>

#include "tests/test.h"

static int expect_map(const char *native, yvex_tensor_role role, const char *target)
{
    char out[256];
    yvex_tensor_role got_role;
    yvex_weight_mapping_issue_kind issue;

    YVEX_TEST_ASSERT(yvex_gguf_map_deepseek_name(native, out, sizeof(out), &got_role, &issue) == 1,
                     "native name maps");
    YVEX_TEST_ASSERT(got_role == role, "role matches");
    YVEX_TEST_ASSERT(strcmp(out, target) == 0, "target matches");
    YVEX_TEST_ASSERT(issue == YVEX_WEIGHT_MAPPING_ISSUE_NONE, "no mapping issue");
    return 0;
}

static int test_deepseek_patterns(void)
{
    if (expect_map("embed.weight", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, "token_embd.weight") != 0) return 1;
    if (expect_map("model.layers.3.self_attn.q_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_Q, "blk.3.attn_q.weight") != 0) return 1;
    if (expect_map("model.layers.3.self_attn.k_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_K, "blk.3.attn_k.weight") != 0) return 1;
    if (expect_map("model.layers.3.self_attn.v_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_V, "blk.3.attn_v.weight") != 0) return 1;
    if (expect_map("model.layers.3.self_attn.o_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_OUT, "blk.3.attn_output.weight") != 0) return 1;
    if (expect_map("model.layers.3.input_layernorm.weight", YVEX_TENSOR_ROLE_ATTENTION_NORM, "blk.3.attn_norm.weight") != 0) return 1;
    if (expect_map("model.layers.3.post_attention_layernorm.weight", YVEX_TENSOR_ROLE_FFN_NORM, "blk.3.ffn_norm.weight") != 0) return 1;
    if (expect_map("model.layers.3.mlp.gate_proj.weight", YVEX_TENSOR_ROLE_FFN_GATE, "blk.3.ffn_gate.weight") != 0) return 1;
    if (expect_map("model.layers.3.mlp.up_proj.weight", YVEX_TENSOR_ROLE_FFN_UP, "blk.3.ffn_up.weight") != 0) return 1;
    if (expect_map("model.layers.3.mlp.down_proj.weight", YVEX_TENSOR_ROLE_FFN_DOWN, "blk.3.ffn_down.weight") != 0) return 1;
    if (expect_map("model.layers.3.mlp.gate.weight", YVEX_TENSOR_ROLE_MOE_ROUTER, "blk.3.ffn_gate_inp.weight") != 0) return 1;
    if (expect_map("model.layers.3.mlp.experts.7.gate_proj.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, "blk.3.ffn.experts.7.gate.weight") != 0) return 1;
    if (expect_map("model.layers.3.mlp.experts.7.up_proj.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_UP, "blk.3.ffn.experts.7.up.weight") != 0) return 1;
    if (expect_map("model.layers.3.mlp.experts.7.down_proj.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, "blk.3.ffn.experts.7.down.weight") != 0) return 1;
    if (expect_map("blk.3.attn_q.weight", YVEX_TENSOR_ROLE_ATTENTION_Q, "blk.3.attn_q.weight") != 0) return 1;
    if (expect_map("layers.3.attn_norm.weight", YVEX_TENSOR_ROLE_ATTENTION_NORM, "blk.3.attn_norm.weight") != 0) return 1;
    if (expect_map("layers.3.ffn_norm.weight", YVEX_TENSOR_ROLE_FFN_NORM, "blk.3.ffn_norm.weight") != 0) return 1;
    if (expect_map("layers.3.ffn.gate.weight", YVEX_TENSOR_ROLE_MOE_ROUTER, "blk.3.ffn_gate_inp.weight") != 0) return 1;
    if (expect_map("layers.3.ffn.experts.7.w1.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, "blk.3.ffn.experts.7.gate.weight") != 0) return 1;
    if (expect_map("layers.3.ffn.experts.7.w2.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, "blk.3.ffn.experts.7.down.weight") != 0) return 1;
    if (expect_map("layers.3.ffn.experts.7.w3.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_UP, "blk.3.ffn.experts.7.up.weight") != 0) return 1;
    return 0;
}

static int test_unknown(void)
{
    char out[256];
    yvex_tensor_role role;
    yvex_weight_mapping_issue_kind issue;

    YVEX_TEST_ASSERT(yvex_gguf_map_deepseek_name("unknown.weight", out, sizeof(out), &role, &issue) == 0,
                     "unknown native name unmapped");
    YVEX_TEST_ASSERT(role == YVEX_TENSOR_ROLE_UNKNOWN, "unknown role");
    YVEX_TEST_ASSERT(issue == YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME, "unknown native issue");
    return 0;
}

int yvex_test_deepseek_adapter(void)
{
    if (test_deepseek_patterns() != 0) return 1;
    if (test_unknown() != 0) return 1;
    return 0;
}
