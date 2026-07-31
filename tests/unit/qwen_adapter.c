
#include <string.h>

#include <yvex/gguf.h>

#include "tests/test.h"

static int maps(const char *name, yvex_tensor_role role, const char *target)
{
    char buf[128];
    yvex_tensor_role out_role;
    yvex_weight_mapping_issue_kind issue;
    int rc = yvex_qwen_adapter_map_name(name, buf, sizeof(buf), &out_role, &issue);
    return rc && out_role == role && strcmp(buf, target) == 0;
}

int yvex_test_qwen_adapter(void)
{
    char buf[128];
    yvex_tensor_role role;
    yvex_weight_mapping_issue_kind issue;

    YVEX_TEST_ASSERT(maps("model.embed_tokens.weight", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, "token_embd.weight"), "embed");
    YVEX_TEST_ASSERT(maps("model.layers.1.self_attn.q_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_Q, "blk.1.attn_q.weight"), "q");
    YVEX_TEST_ASSERT(maps("model.layers.1.self_attn.k_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_K, "blk.1.attn_k.weight"), "k");
    YVEX_TEST_ASSERT(maps("model.layers.1.self_attn.v_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_V, "blk.1.attn_v.weight"), "v");
    YVEX_TEST_ASSERT(maps("model.layers.1.self_attn.o_proj.weight", YVEX_TENSOR_ROLE_ATTENTION_OUT, "blk.1.attn_output.weight"), "o");
    YVEX_TEST_ASSERT(maps("model.layers.1.input_layernorm.weight", YVEX_TENSOR_ROLE_ATTENTION_NORM, "blk.1.attn_norm.weight"), "attn norm");
    YVEX_TEST_ASSERT(maps("model.layers.1.post_attention_layernorm.weight", YVEX_TENSOR_ROLE_FFN_NORM, "blk.1.ffn_norm.weight"), "ffn norm");
    YVEX_TEST_ASSERT(maps("model.layers.1.mlp.gate_proj.weight", YVEX_TENSOR_ROLE_FFN_GATE, "blk.1.ffn_gate.weight"), "gate");
    YVEX_TEST_ASSERT(maps("model.layers.1.mlp.up_proj.weight", YVEX_TENSOR_ROLE_FFN_UP, "blk.1.ffn_up.weight"), "up");
    YVEX_TEST_ASSERT(maps("model.layers.1.mlp.down_proj.weight", YVEX_TENSOR_ROLE_FFN_DOWN, "blk.1.ffn_down.weight"), "down");
    YVEX_TEST_ASSERT(maps("model.norm.weight", YVEX_TENSOR_ROLE_OUTPUT_NORM, "output_norm.weight"), "norm");
    YVEX_TEST_ASSERT(maps("lm_head.weight", YVEX_TENSOR_ROLE_OUTPUT_HEAD, "output.weight"), "head");
    YVEX_TEST_ASSERT(!yvex_qwen_adapter_map_name("unknown.weight", buf, sizeof(buf), &role, &issue), "unknown");
    return 0;
}
