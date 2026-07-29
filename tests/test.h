/*
 * YVEX - Test helpers
 *
 * File: tests/test.h
 * Layer: test
 *
 * Purpose:
 *   Provides tiny assertion helpers for core C tests. This is intentionally
 *   small and dependency-free.
 *
 * Covers:
 *   - failure reporting to stderr
 *   - boolean assertions
 *   - string equality assertions
 *
 * Commands:
 *   - make test-core
 *
 * Expected:
 *   - tests exit 0 on success
 *   - tests print concise failure messages to stderr
 */
#ifndef YVEX_TEST_H
#define YVEX_TEST_H

#include <stdio.h>
#include <string.h>

/* Canonical unit-runner registry; every test owner includes this harness. */
int yvex_test_artifact(void);
int yvex_test_artifact_integrity(void);
int yvex_test_artifact_naming(void);
int yvex_test_backend_cpu(void);
int yvex_test_backend_ops(void);
int yvex_test_conversion_payload(void);
int yvex_test_conversion_plan(void);
int yvex_test_deepseek_adapter(void);
int yvex_test_deepseek_arch_ir(void);
int yvex_test_deepseek_attention(void);
int yvex_test_deepseek_tensor_coverage(void);
struct yvex_deepseek_gguf_map;
int yvex_test_deepseek_map_fixture_build(struct yvex_deepseek_gguf_map **out);
int yvex_test_dtype(void);
int yvex_test_error(void);
int yvex_test_fs(void);
int yvex_test_gguf(void);
int yvex_test_gguf_artifact_abi(void);
int yvex_test_gguf_emit(void);
int yvex_test_gguf_layout_integrity(void);
int yvex_test_gguf_qtype_abi(void);
int yvex_test_gguf_template(void);
int yvex_test_gguf_writer_artifact(void);
int yvex_test_graph(void);
int yvex_test_protocol(void);
int yvex_test_imatrix(void);
int yvex_test_log(void);
int yvex_test_materialization_runtime(void);
int yvex_test_materialize_cpu(void);
int yvex_test_materialize_gate(void);
int yvex_test_memory_plan(void);
int yvex_test_model_descriptor(void);
int yvex_test_model_gate(void);
int yvex_test_model_ref(void);
int yvex_test_model_registry(void);
int yvex_test_native_weights(void);
int yvex_test_openai(void);
int yvex_test_planner(void);
int yvex_test_prompt(void);
int yvex_test_provider(void);
int yvex_test_qtype_support(void);
int yvex_test_quant_execute(void);
int yvex_test_quant_job(void);
int yvex_test_quant_numeric(void);
int yvex_test_quant_policy(void);
int yvex_test_qwen_adapter(void);
int yvex_test_runtime_binding(void);
int yvex_test_runtime_decode(void);
int yvex_test_runtime_logits(void);
int yvex_test_runtime_sampling(void);
int yvex_test_runtime_generation(void);
int yvex_test_runtime_tokenizer(void);
int yvex_test_runtime_moe(void);
int yvex_test_runtime_transformer(void);
int yvex_test_runtime_prefill(void);
int yvex_test_runtime_state(void);
int yvex_test_runtime_benchmark(void);
int yvex_test_safetensors_header(void);
int yvex_test_server(void);
int yvex_test_shape(void);
int yvex_test_source_manifest(void);
int yvex_test_source_payload(void);
int yvex_test_source_verify(void);
int yvex_test_status(void);
int yvex_test_tensor_table(void);
int yvex_test_token_input(void);
int yvex_test_tokenizer(void);
int yvex_test_transform_ir(void);
int yvex_test_version(void);
int yvex_test_weight_mapping(void);
int yvex_test_weights(void);

int yvex_cuda_test_info(void);
int yvex_cuda_test_graph(void);
int yvex_cuda_test_materialize_cuda(void);
int yvex_cuda_test_ops(void);
int yvex_cuda_test_parity(void);
int yvex_cuda_test_quant_qtype(void);
int yvex_cuda_test_tensor(void);

#define YVEX_TEST_FAIL(msg) \
    do { \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
        return 1; \
    } while (0)

#define YVEX_TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            YVEX_TEST_FAIL(msg); \
        } \
    } while (0)

#define YVEX_TEST_ASSERT_STREQ(actual, expected, msg) \
    do { \
        const char *yvex_test_actual = (actual); \
        const char *yvex_test_expected = (expected); \
        if (!yvex_test_actual || !yvex_test_expected || strcmp(yvex_test_actual, yvex_test_expected) != 0) { \
            fprintf(stderr, "FAIL: %s:%d: %s: expected '%s', got '%s'\n", \
                    __FILE__, __LINE__, (msg), \
                    yvex_test_expected ? yvex_test_expected : "(null)", \
                    yvex_test_actual ? yvex_test_actual : "(null)"); \
            return 1; \
        } \
    } while (0)

#endif /* YVEX_TEST_H */
