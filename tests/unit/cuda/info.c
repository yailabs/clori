/*
 * YVEX - CUDA info tests
 *
 * File: tests/test_cuda_info.c
 * Layer: test
 *
 * Purpose:
 *   Proves that the CUDA backend CUDA device probe opens a real CUDA backend when the
 *   local driver/device is available. Returns 77 when CUDA is unavailable.
 */
#include <stdio.h>
#include <stdlib.h>

#include <yvex/api.h>

#include "tests/test.h"

/* Contract: checks one exact CUDA variant and its admitted bundle/function facts. */
static int assert_supported_variant(const yvex_backend *backend,
                                    yvex_backend_operation_variant variant)
{
    yvex_backend_capability_result result;
    yvex_error err;
    int rc = yvex_backend_query_capability(backend, variant, &result, &err);

    YVEX_TEST_ASSERT(rc == YVEX_OK, "query exact CUDA capability");
    YVEX_TEST_ASSERT(result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "exact CUDA variant supported");
    YVEX_TEST_ASSERT(result.reason == YVEX_BACKEND_CAPABILITY_REASON_NONE,
                     "supported CUDA variant has no refusal reason");
    YVEX_TEST_ASSERT(result.context_available, "CUDA variant has context");
    if (variant >= YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) {
        YVEX_TEST_ASSERT(result.kernel_bundle_available,
                         "kernel CUDA variant has admitted generated bundle");
        YVEX_TEST_ASSERT(result.function_available,
                         "kernel CUDA variant has resolved function");
    }
    return 0;
}

/* Contract: proves atomic bundle rejection, cleared handles, and clean retry. */
static int assert_bundle_rollback(const char *failure,
                                  yvex_backend_capability_reason expected_reason,
                                  yvex_backend_operation_variant variant)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_capability_result result;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_BUNDLE_FAILURE", failure, 1) == 0,
                     "set bundle failure injection");
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "context survives rejected kernel bundle");
    YVEX_TEST_ASSERT(yvex_backend_status_of(backend) == YVEX_BACKEND_STATUS_CONTEXT_READY,
                     "rejected bundle leaves context-only status");
    rc = yvex_backend_query_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_ALLOC,
                                       &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "Driver memory capability survives bundle rejection");
    rc = yvex_backend_query_capability(backend, variant, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "query rejected kernel capability");
    YVEX_TEST_ASSERT(result.state == YVEX_BACKEND_CAPABILITY_FAILED,
                     "rejected bundle cannot support kernel variant");
    YVEX_TEST_ASSERT(result.reason == expected_reason,
                     "rejected bundle exposes typed reason");
    YVEX_TEST_ASSERT(!result.kernel_bundle_available && !result.function_available,
                     "rejected bundle leaves no admitted handle");
    yvex_backend_close(backend);
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_BUNDLE_FAILURE") == 0,
                     "clear bundle failure injection");
    return 0;
}

int yvex_cuda_test_info(void)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_device_info info;
    yvex_error err;
    static const char *attention_symbols[] = {
        "yvex_attention_bf16_round",
        "yvex_qtype_matvec",
        "yvex_deepseek_decode",
        "yvex_deepseek_weighted_norm",
        "yvex_deepseek_unit_norm",
        "yvex_deepseek_rope",
        "yvex_deepseek_activation",
        "yvex_deepseek_rolling",
        "yvex_deepseek_topk",
        "yvex_deepseek_reduce"
    };
    size_t symbol_index;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cuda backend");
    rc = yvex_backend_get_device_info(backend, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "get cuda device info");
    YVEX_TEST_ASSERT(info.kind == YVEX_BACKEND_KIND_CUDA, "device info kind");
    YVEX_TEST_ASSERT(info.name && info.name[0] != '\0', "device name non-empty");
    YVEX_TEST_ASSERT(info.compute_capability_major >= 1, "compute capability major");
    YVEX_TEST_ASSERT(info.global_memory_bytes > 0, "global memory nonzero");
    YVEX_TEST_ASSERT(info.total_memory_bytes > 0, "total memory nonzero");

    for (rc = 0; rc < (int)YVEX_BACKEND_VARIANT_COUNT; ++rc) {
        YVEX_TEST_ASSERT(assert_supported_variant(
                             backend, (yvex_backend_operation_variant)rc) == 0,
                         "all advertised CUDA variants are exact");
    }
    yvex_backend_close(backend);

    YVEX_TEST_ASSERT(assert_bundle_rollback(
                         "module",
                         YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED,
                         YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "module failure rolls back atomically");
    YVEX_TEST_ASSERT(assert_bundle_rollback(
                         "symbol",
                         YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
                         YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "symbol failure rolls back atomically");
    for (symbol_index = 0u;
         symbol_index < sizeof(attention_symbols) / sizeof(attention_symbols[0]);
         ++symbol_index) {
        YVEX_TEST_ASSERT(assert_bundle_rollback(
                             attention_symbols[symbol_index],
                             YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
                             YVEX_BACKEND_VARIANT_ATTENTION_ENCODED) == 0,
                         "each encoded-attention symbol is atomically required");
    }

    backend = NULL;
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "CUDA bundle admission retries after rollback");
    YVEX_TEST_ASSERT(assert_supported_variant(
                         backend, YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "retry resolves canonical function");
    yvex_backend_close(backend);
    return 0;
}
