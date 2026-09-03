/* Qualify pinned Qwen3.5 hybrid semantics without model weights. */
#include "tests/test.h"

#include <yvex/internal/core.h>
#include <yvex/internal/families/qwen3_5.h>
#include <yvex/internal/graph.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static int qwen_test_dir(const char *path)
{
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

static int qwen_test_config(const char *path, unsigned long long interval)
{
    FILE *fp = fopen(path, "wb");
    unsigned long long layer;

    if (!fp) return 0;
    if (fprintf(
            fp,
            "{\"architectures\":[\"Qwen3_5ForConditionalGeneration\"],"
            "\"image_token_id\":248056,\"language_model_only\":false,"
            "\"model_type\":\"qwen3_5\",\"text_config\":{"
            "\"attention_bias\":false,\"attention_dropout\":0.0,"
            "\"attn_output_gate\":true,\"bos_token_id\":248044,"
            "\"dtype\":\"bfloat16\",\"eos_token_id\":248044,"
            "\"full_attention_interval\":%llu,\"head_dim\":256,"
            "\"hidden_act\":\"silu\",\"hidden_size\":5120,"
            "\"intermediate_size\":17408,\"layer_types\":[",
            interval) < 0) {
        fclose(fp);
        return 0;
    }
    for (layer = 0ull; layer < 64ull; ++layer) {
        const char *kind = (layer + 1ull) % 4ull == 0ull
                               ? "full_attention"
                               : "linear_attention";

        if (fprintf(fp, "%s\"%s\"", layer ? "," : "", kind) < 0) {
            fclose(fp);
            return 0;
        }
    }
    if (fprintf(
            fp,
            "],\"linear_conv_kernel_dim\":4,\"linear_key_head_dim\":128,"
            "\"linear_num_key_heads\":16,\"linear_num_value_heads\":48,"
            "\"linear_value_head_dim\":128,\"mamba_ssm_dtype\":\"float32\","
            "\"max_position_embeddings\":262144,"
            "\"model_type\":\"qwen3_5_text\",\"mtp_num_hidden_layers\":1,"
            "\"mtp_use_dedicated_embeddings\":false,"
            "\"num_attention_heads\":24,\"num_hidden_layers\":64,"
            "\"num_key_value_heads\":4,\"output_gate_type\":\"swish\","
            "\"pad_token_id\":null,\"partial_rotary_factor\":0.25,"
            "\"rms_norm_eps\":0.000001,\"rope_parameters\":{"
            "\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],"
            "\"partial_rotary_factor\":0.25,\"rope_theta\":10000000,"
            "\"rope_type\":\"default\"},\"tie_word_embeddings\":false,"
            "\"use_cache\":true,\"vocab_size\":248320},"
            "\"tie_word_embeddings\":false,"
            "\"transformers_version\":\"5.8.0.dev0\","
            "\"video_token_id\":248057,\"vision_config\":{"
            "\"deepstack_visual_indexes\":[],\"depth\":27,"
            "\"hidden_act\":\"gelu_pytorch_tanh\",\"hidden_size\":1152,"
            "\"intermediate_size\":4304,\"model_type\":\"qwen3_5\","
            "\"num_heads\":16,\"num_position_embeddings\":2304,"
            "\"out_hidden_size\":5120,\"patch_size\":16,"
            "\"spatial_merge_size\":2,\"temporal_patch_size\":2},"
            "\"vision_end_token_id\":248054,\"vision_start_token_id\":248053}") < 0) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static int qwen_test_generation(const char *path)
{
    FILE *fp = fopen(path, "wb");

    if (!fp) return 0;
    if (fputs("{\"bos_token_id\":248044,\"do_sample\":true,"
              "\"eos_token_id\":[248046,248044],\"pad_token_id\":248044,"
              "\"temperature\":1.0,\"top_k\":20,\"top_p\":0.95}", fp) < 0) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static void qwen_test_verification(yvex_source_verification *verification,
                                   const char *root)
{
    memset(verification, 0, sizeof(*verification));
    verification->verified = 1;
    verification->config_valid = 1;
    yvex_core_text_copy(verification->resolved_source_path,
                        sizeof(verification->resolved_source_path), root);
    yvex_core_text_copy(verification->repository_id,
                        sizeof(verification->repository_id),
                        "Qwen/Qwen3.8-27B");
    yvex_core_text_copy(verification->revision,
                        sizeof(verification->revision),
                        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0");
    yvex_core_text_copy(verification->model_type,
                        sizeof(verification->model_type), "qwen3_5");
    yvex_core_text_copy(verification->architecture,
                        sizeof(verification->architecture),
                        "Qwen3_5ForConditionalGeneration");
}

int yvex_test_qwen3_5_architecture(void)
{
    const char *root = "build/tests/qwen3-5-architecture";
    const yvex_qwen3_5_api *api = yvex_model_register_qwen3_5();
    const yvex_graph_execution_binding *execution =
        yvex_graph_execution_find(0ull, 0ull, YVEX_QWEN3_8_27B_TARGET_ID);
    const yvex_qwen3_5_architecture *architecture;
    yvex_qwen3_5_model *model = NULL;
    yvex_qwen3_5_failure failure;
    yvex_source_verification verification;
    yvex_error err;
    char config[512], generation[512];
    int rc;

    YVEX_TEST_ASSERT(qwen_test_dir("build") && qwen_test_dir("build/tests") &&
                         qwen_test_dir(root),
                     "create Qwen semantic fixture root");
    snprintf(config, sizeof(config), "%s/config.json", root);
    snprintf(generation, sizeof(generation), "%s/generation_config.json", root);
    YVEX_TEST_ASSERT(qwen_test_config(config, 4ull) &&
                         qwen_test_generation(generation),
                     "write pinned-shape Qwen configuration fixture");
    qwen_test_verification(&verification, root);
    yvex_error_clear(&err);
    rc = api ? api->open(&model, &verification, &failure, &err)
             : YVEX_ERR_STATE;
    if (rc != YVEX_OK)
        fprintf(stderr, "Qwen architecture refusal: %s field=%s reason=%s\n",
                yvex_error_message(&err), failure.field,
                failure.reason ? failure.reason : "none");
    YVEX_TEST_ASSERT(api && api->schema_version == 3u &&
                         rc == YVEX_OK,
                     "open authenticated Qwen3.5 semantic architecture");
    YVEX_TEST_ASSERT(
        execution && execution->compiler &&
            !strcmp(execution->logical_transform_identity,
                    YVEX_QWEN3_5_LOGICAL_TRANSFORM_IDENTITY) &&
            !strcmp(execution->compiler->logical_transform_identity,
                    YVEX_QWEN3_5_LOGICAL_TRANSFORM_IDENTITY),
        "Qwen deployment and compiler expose one current logical transform identity");
    architecture = api->architecture(model);
    YVEX_TEST_ASSERT(
        architecture && strcmp(architecture->product_id, "qwen3.8-27b") == 0 &&
            strcmp(architecture->semantic_family, "qwen3_5") == 0 &&
            architecture->source_multimodal && architecture->text_specialization &&
            architecture->vision_execution_deferred &&
            architecture->mtp_acceleration_deferred &&
            yvex_sha256_hex_valid(architecture->architecture_identity),
        "release, architecture family, specialization, and identity remain distinct");
    YVEX_TEST_ASSERT(
        architecture->text.hidden_size == 5120ull &&
            architecture->text.layer_count == 64ull &&
            architecture->text.linear_attention_layers == 48ull &&
            architecture->text.full_attention_layers == 16ull &&
            architecture->text.intermediate_size == 17408ull &&
            architecture->text.vocabulary_size == 248320ull &&
            architecture->text.maximum_positions == 262144ull &&
            architecture->text.rotary_dimension == 64ull &&
            architecture->text.recurrent_state_f32 &&
            api->layer_kind(model, 2ull) ==
                YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION &&
            api->layer_kind(model, 3ull) ==
                YVEX_QWEN3_5_LAYER_FULL_ATTENTION,
        "pinned hybrid text topology is exact");
    YVEX_TEST_ASSERT(
        architecture->text.attention_heads == 24ull &&
            architecture->text.kv_heads == 4ull &&
            architecture->text.attention_head_dimension == 256ull &&
            architecture->text.linear_key_heads == 16ull &&
            architecture->text.linear_value_heads == 48ull &&
            architecture->text.linear_key_head_dimension == 128ull &&
            architecture->text.linear_value_head_dimension == 128ull &&
            architecture->text.linear_convolution_kernel == 4ull,
        "full-attention and recurrent sequence-mixer geometry is exact");
    YVEX_TEST_ASSERT(
        architecture->vision.depth == 27ull &&
            architecture->vision.hidden_size == 1152ull &&
            architecture->vision.output_hidden_size == 5120ull &&
            architecture->generation.stop_token_count == 2ull &&
            architecture->generation.stop_token_ids[0] == 248046ull &&
            architecture->generation.stop_token_ids[1] == 248044ull,
        "deferred vision and generation facts remain accounted");
    api->close(&model);
    YVEX_TEST_ASSERT(!model, "close semantic architecture");

    YVEX_TEST_ASSERT(qwen_test_config(config, 5ull),
                     "write inconsistent hybrid interval fixture");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        api->open(&model, &verification, &failure, &err) != YVEX_OK && !model &&
            failure.code == YVEX_QWEN3_5_FAILURE_CONFIGURATION &&
            strcmp(failure.field, "layer_types") == 0,
        "inconsistent hybrid topology fails closed");
    verification.revision[0] = '0';
    YVEX_TEST_ASSERT(
        api->open(&model, &verification, &failure, &err) != YVEX_OK &&
            failure.code == YVEX_QWEN3_5_FAILURE_SOURCE_IDENTITY,
        "mutable or foreign source identity cannot construct Qwen semantics");
    return 0;
}
