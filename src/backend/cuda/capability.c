/*
 * Admit the exact generated kernel bundle and mediate fail-closed CUDA launches.
 *
 * No kernel variant is supported before the canonical generated bundle and every required function
 * are admitted; failed launch or synchronization never remains a supported variant. Bounded
 * primitive capability is not transformer or generation support.
 */
#include "src/backend/cuda/private.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef YVEX_HAVE_CUDA_KERNEL_PTX
#include <cuda_kernels_ptx.inc>
#endif
#ifdef YVEX_HAVE_CUDA_KERNEL_CUBIN
#include <cuda_kernels_cubin.inc>
#endif
typedef struct {
    const char *symbol;
    yvex_backend_operation_variant variant;
    size_t state_offset;
} cuda_kernel_binding;
#define CUDA_HANDLE_OFFSET(field) offsetof(yvex_cuda_backend_state, field)
/*
 * The bundle table is the single correspondence between generated PTX names,
 * capability variants, and admitted state.  Aliased variants (dense/routed
 * MLP and causal/non-causal attention) intentionally share one binding. */
static const cuda_kernel_binding cuda_kernel_bindings[] = {
    {"yvex_embed_f32", YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32,
     CUDA_HANDLE_OFFSET(embed_function)},
    {"yvex_embed_f16_to_f32", YVEX_BACKEND_VARIANT_EMBED_F16_TO_F32,
     CUDA_HANDLE_OFFSET(embed_f16_function)},
    {"yvex_rms_norm_f32_weight_f32", YVEX_BACKEND_VARIANT_RMS_NORM_F32_WEIGHT_F32,
     CUDA_HANDLE_OFFSET(rms_norm_f32_function)},
    {"yvex_rms_norm_f32_weight_f16", YVEX_BACKEND_VARIANT_RMS_NORM_F32_WEIGHT_F16,
     CUDA_HANDLE_OFFSET(rms_norm_f16_function)},
    {"yvex_rms_norm_bf16_policy_f32", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(rms_norm_bf16_policy_function)},
    {"yvex_rope_f32", YVEX_BACKEND_VARIANT_ROPE_F32, CUDA_HANDLE_OFFSET(rope_function)},
    {"yvex_matmul_f32", YVEX_BACKEND_VARIANT_MATMUL_F32,
     CUDA_HANDLE_OFFSET(matmul_function)},
    {"yvex_qtype_row_dot", YVEX_BACKEND_VARIANT_QTYPE_ROW_DOT,
     CUDA_HANDLE_OFFSET(qtype_row_dot_function)},
    {"yvex_attention_bf16_round", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(attention_bf16_round_function)},
    {"yvex_f32_to_bf16", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(bf16_pack_function)},
    {"yvex_qtype_matvec", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(qtype_matvec_function)},
    {"yvex_qtype_split_matvec", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(qtype_split_matvec_function)},
    {"yvex_qtype_gather", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(qtype_gather_function)},
    {"yvex_argmax_f32", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(argmax_f32_function)},
    {"yvex_sample_stochastic_f32", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(sample_stochastic_f32_function)},
    {"yvex_speculation_stochastic_f32", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(speculation_stochastic_f32_function)},
    {"yvex_q8_quantize", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(q8_quantize_function)},
    {"yvex_deepseek_decode", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_decode_function)},
    {"yvex_deepseek_weighted_norm", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_weighted_norm_function)},
    {"yvex_deepseek_unit_norm", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_unit_norm_function)},
    {"yvex_deepseek_rope", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_rope_function)},
    {"yvex_deepseek_activation", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_activation_function)},
    {"yvex_deepseek_mhc_pre", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_mhc_pre_function)},
    {"yvex_deepseek_mhc_post", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_mhc_post_function)},
    {"yvex_transformer_feature_mean", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(transformer_feature_mean_function)},
    {"yvex_deepseek_transformer_final", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_transformer_final_function)},
    {"yvex_deepseek_rolling", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_rolling_function)},
    {"yvex_deepseek_topk", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_topk_function)},
    {"yvex_deepseek_reduce", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_reduce_function)},
    {"yvex_moe_route", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_route_function)},
    {"yvex_moe_route_rows", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_route_rows_function)},
    {"yvex_moe_pair_order", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_pair_order_function)},
    {"yvex_moe_grouped_up", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_grouped_up_function)},
    {"yvex_moe_grouped_down", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_grouped_down_function)},
    {"yvex_moe_grouped_up_rows", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_grouped_up_rows_function)},
    {"yvex_moe_grouped_down_rows", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_grouped_down_rows_function)},
    {"yvex_moe_reduce_rows", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_reduce_rows_function)},
    {"yvex_moe_combine_rows", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_combine_rows_function)},
    {"yvex_moe_swiglu", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_swiglu_function)},
    {"yvex_moe_accumulate", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_accumulate_function)},
    {"yvex_mlp_f32", YVEX_BACKEND_VARIANT_MLP_DENSE_F32,
     CUDA_HANDLE_OFFSET(mlp_function)},
    {"yvex_attention_f32", YVEX_BACKEND_VARIANT_ATTENTION_CAUSAL_F32,
     CUDA_HANDLE_OFFSET(attention_function)},
    {"yvex_rotary_half_f32", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(rotary_half_function)},
    {"yvex_gqa_causal_f32", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(gqa_causal_function)},
    {"yvex_silu_product_bf16_f32", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(silu_product_function)},
};
#define CUDA_KERNEL_BINDING_COUNT (sizeof(cuda_kernel_bindings) / sizeof(cuda_kernel_bindings[0]))
#undef CUDA_HANDLE_OFFSET

static CUfunction *cuda_function_slot(yvex_cuda_backend_state *state, size_t offset)
{
    return (CUfunction *)((unsigned char *)state + offset);
}

static void cuda_bundle_clear_handles(yvex_cuda_backend_state *state)
{
    size_t index;
    if (!state) {
        return;
    }
    memset(state->modules, 0, sizeof(state->modules));
    state->module_count = 0u;
    for (index = 0; index < CUDA_KERNEL_BINDING_COUNT; ++index) {
        *cuda_function_slot(state, cuda_kernel_bindings[index].state_offset) = NULL;
    }
    state->kernel_bundle_native = 0;
    state->kernel_bundle_identity[0] = '\0';
    state->kernel_bundle_architecture[0] = '\0';
}
#ifdef YVEX_HAVE_CUDA_KERNEL_PTX

typedef struct {
    const unsigned char *const *images;
    const unsigned long long *image_bytes;
    size_t image_count;
    const char *architecture;
    int native;
} cuda_bundle_image;

static int cuda_bundle_select_image(const yvex_backend *backend,
                                    cuda_bundle_image *image,
                                    yvex_error *err)
{
    const char *forced = getenv("YVEX_TEST_CUDA_BUNDLE_IMAGE");
    char device_architecture[16];
    if (!backend || !image) return YVEX_ERR_INVALID_ARG;
    memset(image, 0, sizeof(*image));
    (void)snprintf(device_architecture, sizeof(device_architecture), "sm_%d%d",
                   backend->device_info.compute_capability_major,
                   backend->device_info.compute_capability_minor);
#ifdef YVEX_HAVE_CUDA_KERNEL_CUBIN
    if ((!forced || strcmp(forced, "ptx") != 0) &&
        strcmp(device_architecture, cuda_kernels_cubin_arch) == 0) {
        image->images = cuda_kernel_cubin_images;
        image->image_bytes = cuda_kernel_cubin_image_bytes;
        image->image_count = CUDA_KERNEL_CUBIN_IMAGE_COUNT;
        image->architecture = cuda_kernels_cubin_arch;
        image->native = 1;
        if (image->image_count && image->image_count <= YVEX_CUDA_KERNEL_MODULE_MAX)
            return YVEX_OK;
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.kernels.image",
                       "native CUDA module count exceeds the admitted bundle capacity");
        return YVEX_ERR_BOUNDS;
    }
#endif
    if (forced && strcmp(forced, "native") == 0) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "cuda.kernels.image",
                        "native CUDA image is unavailable for %s", device_architecture);
        return YVEX_ERR_UNSUPPORTED;
    }
    image->images = cuda_kernel_ptx_images;
    image->image_bytes = cuda_kernel_ptx_image_bytes;
    image->image_count = CUDA_KERNEL_PTX_IMAGE_COUNT;
    image->architecture = "ptx";
    if (image->image_count && image->image_count <= YVEX_CUDA_KERNEL_MODULE_MAX)
        return YVEX_OK;
    yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.kernels.image",
                   "PTX CUDA module count exceeds the admitted bundle capacity");
    return YVEX_ERR_BOUNDS;
}

static int cuda_bundle_identity(const cuda_bundle_image *image,
                                char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    size_t index;
    if (!image || !image->images || !image->image_bytes || !image->image_count ||
        image->image_count > YVEX_CUDA_KERNEL_MODULE_MAX || !image->architecture)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.cuda.kernel-bundle.v3") ||
        !yvex_sha256_update_text(&hash, image->native ? "native" : "ptx") ||
        !yvex_sha256_update_text(&hash, image->architecture) ||
        !yvex_sha256_update_u64(&hash, image->image_count))
        return 0;
    for (index = 0; index < image->image_count; ++index)
        if (!image->images[index] || !image->image_bytes[index] ||
            !yvex_sha256_update_u64(&hash, index) ||
            !yvex_sha256_update_u64(&hash, image->image_bytes[index]) ||
            !yvex_sha256_update(&hash, image->images[index],
                                (size_t)image->image_bytes[index]))
            return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
#endif

static int cuda_test_failure_matches(const char *name,
                                     yvex_backend_operation_variant variant)
{
    const char *value = getenv(name);
    return value && value[0] != '\0' &&
           (strcmp(value, "all") == 0 ||
            strcmp(value, yvex_backend_operation_variant_name(variant)) == 0);
}

static CUfunction cuda_variant_function(const yvex_cuda_backend_state *state,
                                        yvex_backend_operation_variant variant)
{
    size_t index;
    CUfunction first = NULL;
    if (!state) {
        return NULL;
    }
    if (variant == YVEX_BACKEND_VARIANT_MLP_ROUTED_F32) {
        variant = YVEX_BACKEND_VARIANT_MLP_DENSE_F32;
    } else if (variant == YVEX_BACKEND_VARIANT_ATTENTION_NONCAUSAL_F32) {
        variant = YVEX_BACKEND_VARIANT_ATTENTION_CAUSAL_F32;
    }
    for (index = 0; index < CUDA_KERNEL_BINDING_COUNT; ++index) {
        if (cuda_kernel_bindings[index].variant == variant) {
            CUfunction function = *cuda_function_slot(
                (yvex_cuda_backend_state *)state,
                cuda_kernel_bindings[index].state_offset);
            if (!function) return NULL;
            if (!first) first = function;
        }
    }
    return first;
}

static int cuda_capability_publish(yvex_backend_capability_result *out,
                                   yvex_backend_capability_state state,
                                   yvex_backend_capability_reason reason,
                                   int function_available, yvex_error *err)
{
    out->state = state;
    out->reason = reason;
    out->function_available = function_available;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void cuda_capability_fail(yvex_backend *backend,
                                 yvex_backend_operation_variant variant,
                                 yvex_backend_capability_reason reason)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    if (!state || variant < 0 || variant >= YVEX_BACKEND_VARIANT_COUNT) {
        return;
    }
    state->variant_failures[variant] = reason;
    if (reason == YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED ||
        reason == YVEX_BACKEND_CAPABILITY_REASON_CLEANUP_FAILED) {
        state->backend_failure_reason = reason;
        backend->status = YVEX_BACKEND_STATUS_FAILED;
    }
}
#ifdef YVEX_HAVE_CUDA_KERNEL_PTX

static void cuda_bundle_retain_rejected(yvex_backend *backend,
                                        const CUmodule *modules,
                                        unsigned int module_count)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    cuda_bundle_clear_handles(state);
    if (!state || !modules || !module_count ||
        module_count > YVEX_CUDA_KERNEL_MODULE_MAX) return;
    memcpy(state->modules, modules, module_count * sizeof(*modules));
    state->module_count = module_count;
}
/* Contract: resolves one required function or returns a typed atomic-admission failure. */

static int cuda_resolve_required(yvex_cuda_backend_state *state,
                                 const CUmodule *modules,
                                 unsigned int module_count,
                                 const char *symbol,
                                 yvex_backend_operation_variant variant,
                                 CUfunction *out,
                                 yvex_error *err)
{
    const char *injected = getenv("YVEX_TEST_CUDA_BUNDLE_FAILURE");
    unsigned int index;
    *out = NULL;
    if (injected &&
        (strcmp(injected, "symbol") == 0 || strcmp(injected, symbol) == 0 ||
         strcmp(injected, yvex_backend_operation_variant_name(variant)) == 0)) {
        yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.kernels.resolve",
                        "required CUDA function unavailable: %s", symbol);
        return YVEX_ERR_BACKEND;
    }
    for (index = 0u; index < module_count; ++index) {
        CUresult status = state->driver.cuModuleGetFunction(out, modules[index], symbol);
        if (status == YVEX_CUDA_SUCCESS) return YVEX_OK;
        if (status != YVEX_CUDA_ERROR_NOT_FOUND)
            return yvex_cuda_status(&state->driver, status,
                                    "cuda.kernels.resolve", err);
    }
    yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.kernels.resolve",
                    "required CUDA function unavailable: %s", symbol);
    return YVEX_ERR_BACKEND;
}
#endif
/*
 * Contract: admits only generated manifest-owned CUDA modules and commits no handle
 * until module load and every required symbol succeed. No tensor payload IO. */

int yvex_cuda_kernel_bundle_admit(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int admit_rc;
    if (!backend || !state || !state->context) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.kernels.admit",
                       "CUDA context is required for kernel admission");
        return YVEX_ERR_STATE;
    }
    admit_rc = backend_dispatch_admit(backend, "cuda.kernels.admit", err);
    if (admit_rc != YVEX_OK) return admit_rc;
    if (state->module_count) {
        if (state->kernel_bundle_state == YVEX_CUDA_KERNEL_BUNDLE_ADMITTED) {
            yvex_error_clear(err);
            return YVEX_OK;
        }
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.kernels.admit",
                       "retained CUDA module ownership requires checked cleanup");
        return YVEX_ERR_STATE;
    }
    cuda_bundle_clear_handles(state);
    state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_ABSENT;
    state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_ABSENT;
    state->kernel_bundle_failure_variant = YVEX_BACKEND_VARIANT_COUNT;
#ifndef YVEX_HAVE_CUDA_KERNEL_PTX
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.kernels.admit",
                   "canonical CUDA kernel bundle was not built");
    return YVEX_ERR_UNSUPPORTED;
#else
    {
        CUmodule modules[YVEX_CUDA_KERNEL_MODULE_MAX] = {0};
        CUfunction functions[CUDA_KERNEL_BINDING_COUNT];
        cuda_bundle_image image;
        const char *injected = getenv("YVEX_TEST_CUDA_BUNDLE_FAILURE");
        size_t index;
        unsigned int loaded = 0u;
        int rc;
        memset(functions, 0, sizeof(functions));
        rc = yvex_cuda_set_current(backend, "cuda.kernels.admit", err);
        if (rc != YVEX_OK) {
            state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
            state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_CONTEXT_UNAVAILABLE;
            return rc;
        }
        rc = cuda_bundle_select_image(backend, &image, err);
        if (rc != YVEX_OK) {
            state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
            state->kernel_bundle_reason =
                YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED;
            return rc;
        }
        for (loaded = 0u; loaded < image.image_count; ++loaded) {
            if (injected && strcmp(injected, "module") == 0) {
                yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.kernels.load",
                               "injected CUDA module admission failure");
                rc = YVEX_ERR_BACKEND;
            } else {
                rc = yvex_cuda_status(
                    &state->driver,
                    state->driver.cuModuleLoadData(&modules[loaded],
                                                   image.images[loaded]),
                    "cuda.kernels.load", err);
            }
            if (rc != YVEX_OK) {
                state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
                state->kernel_bundle_reason =
                    YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED;
                goto reject;
            }
        }
        for (index = 0; index < CUDA_KERNEL_BINDING_COUNT; ++index) {
            const cuda_kernel_binding *binding = &cuda_kernel_bindings[index];
            rc = cuda_resolve_required(state, modules, loaded, binding->symbol,
                                       binding->variant, &functions[index], err);
            if (rc != YVEX_OK) {
                state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
                state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING;
                state->kernel_bundle_failure_variant = binding->variant;
                goto reject;
            }
        }
        memcpy(state->modules, modules, loaded * sizeof(*modules));
        state->module_count = loaded;
        for (index = 0; index < CUDA_KERNEL_BINDING_COUNT; ++index) {
            *cuda_function_slot(state, cuda_kernel_bindings[index].state_offset) = functions[index];
        }
        state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_ADMITTED;
        state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_NONE;
        state->kernel_bundle_failure_variant = YVEX_BACKEND_VARIANT_COUNT;
        state->kernel_bundle_native = image.native;
        yvex_core_text_copy(state->kernel_bundle_architecture,
                            sizeof(state->kernel_bundle_architecture),
                            image.architecture);
        if (!cuda_bundle_identity(&image, state->kernel_bundle_identity)) {
            state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
            state->kernel_bundle_reason =
                YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED;
            yvex_error_set(err, YVEX_ERR_STATE, "cuda.kernels.identity",
                           "generated CUDA kernel bundle identity failed");
            rc = YVEX_ERR_STATE;
            goto reject;
        }
        yvex_error_clear(err);
        return YVEX_OK;
reject:
        cuda_bundle_retain_rejected(backend, modules, loaded);
        return rc;
    }
#endif
}

int yvex_cuda_kernel_bundle_close(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned int index;
    int rc = YVEX_OK;
    if (!backend || !state) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (state->module_count && !state->driver.cuModuleUnload) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "cuda.kernels.unload",
                       "CUDA module unload function is unavailable");
    }
    for (index = 0u; rc == YVEX_OK && index < state->module_count; ++index) {
        if (!state->modules[index]) continue;
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuModuleUnload(state->modules[index]),
                              "cuda.kernels.unload", err);
        if (rc == YVEX_OK) state->modules[index] = NULL;
    }
    if (rc != YVEX_OK) {
        state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_CLEANUP_FAILED;
        state->backend_failure_reason = YVEX_BACKEND_CAPABILITY_REASON_CLEANUP_FAILED;
        backend->status = YVEX_BACKEND_STATUS_FAILED;
        return rc;
    }
    cuda_bundle_clear_handles(state);
    state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_ABSENT;
    state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_ABSENT;
    state->backend_failure_reason = YVEX_BACKEND_CAPABILITY_REASON_NONE;
    if (backend->status == YVEX_BACKEND_STATUS_FAILED && !backend_cleanup_only(backend))
        backend->status = YVEX_BACKEND_STATUS_CONTEXT_READY;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_query_capability(const yvex_backend *backend,
                               yvex_backend_operation_variant variant,
                               yvex_backend_capability_result *out,
                               yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_backend_capability_reason reason;
    CUfunction function;
    if (!backend || !state || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.query_capability",
                       "backend, CUDA state, and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    out->context_available = state->context != NULL;
    out->kernel_bundle_available =
        state->kernel_bundle_state == YVEX_CUDA_KERNEL_BUNDLE_ADMITTED;
    if (!state->context)
        return cuda_capability_publish(
            out, YVEX_BACKEND_CAPABILITY_UNSUPPORTED,
            YVEX_BACKEND_CAPABILITY_REASON_CONTEXT_UNAVAILABLE, 0, err);
    if (variant < 0 || variant >= YVEX_BACKEND_VARIANT_COUNT) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.query_capability",
                       "operation variant is out of range");
        return YVEX_ERR_INVALID_ARG;
    }
    if (backend->status == YVEX_BACKEND_STATUS_FAILED &&
        state->backend_failure_reason != YVEX_BACKEND_CAPABILITY_REASON_NONE)
        return cuda_capability_publish(
            out, YVEX_BACKEND_CAPABILITY_FAILED, state->backend_failure_reason, 0, err);
    if (state->variant_failures[variant] != YVEX_BACKEND_CAPABILITY_REASON_NONE)
        return cuda_capability_publish(
            out, YVEX_BACKEND_CAPABILITY_FAILED, state->variant_failures[variant], 0, err);
    if (variant >= YVEX_BACKEND_VARIANT_TENSOR_ALLOC &&
        variant <= YVEX_BACKEND_VARIANT_TENSOR_COPY)
        return cuda_capability_publish(
            out, YVEX_BACKEND_CAPABILITY_SUPPORTED,
            YVEX_BACKEND_CAPABILITY_REASON_NONE, 1, err);
    if (state->kernel_bundle_state == YVEX_CUDA_KERNEL_BUNDLE_ABSENT)
        return cuda_capability_publish(
            out, YVEX_BACKEND_CAPABILITY_UNSUPPORTED,
            YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_ABSENT, 0, err);
    if (state->kernel_bundle_state == YVEX_CUDA_KERNEL_BUNDLE_REJECTED) {
        reason = state->kernel_bundle_reason;
        if (state->kernel_bundle_reason == YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING &&
            state->kernel_bundle_failure_variant != variant)
            reason = YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED;
        return cuda_capability_publish(
            out, YVEX_BACKEND_CAPABILITY_FAILED, reason, 0, err);
    }
    function = cuda_variant_function(state, variant);
    return cuda_capability_publish(
        out, function ? YVEX_BACKEND_CAPABILITY_SUPPORTED : YVEX_BACKEND_CAPABILITY_FAILED,
        function ? YVEX_BACKEND_CAPABILITY_REASON_NONE
                 : YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
        function != NULL, err);
}

int yvex_cuda_require_capability(yvex_backend *backend,
                                 yvex_backend_operation_variant variant,
                                 const char *where,
                                 yvex_error *err)
{
    yvex_backend_capability_result result;
    int rc = yvex_backend_query_capability(backend, variant, &result, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED) {
        return YVEX_OK;
    }
    yvex_error_setf(err,
                    result.state == YVEX_BACKEND_CAPABILITY_FAILED
                        ? YVEX_ERR_BACKEND : YVEX_ERR_UNSUPPORTED,
                    where ? where : "cuda.require_capability",
                    "%s refused: %s",
                    yvex_backend_operation_variant_name(variant),
                    yvex_backend_capability_reason_name(result.reason));
    return result.state == YVEX_BACKEND_CAPABILITY_FAILED
               ? YVEX_ERR_BACKEND : YVEX_ERR_UNSUPPORTED;
}

int yvex_cuda_launch(yvex_backend *backend,
                     yvex_backend_operation_variant variant,
                     CUfunction function,
                     unsigned int grid_x,
                     unsigned int block_x,
                     unsigned int shared_bytes,
                     void **params,
                     const char *where,
                     yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    rc = yvex_cuda_require_capability(backend, variant, where, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (!state || !function) {
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING);
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "admitted CUDA function handle is missing");
        return YVEX_ERR_BACKEND;
    }
    if (cuda_test_failure_matches("YVEX_TEST_CUDA_LAUNCH_FAILURE", variant)) {
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_LAUNCH_FAILED);
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA launch failure");
        return YVEX_ERR_BACKEND;
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuLaunchKernel(function,
                                                       grid_x, 1, 1,
                                                       block_x, 1, 1,
                                                       shared_bytes,
                                                       yvex_cuda_launch_stream(backend),
                                                       params, NULL),
                          where, err);
    if (rc != YVEX_OK) {
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_LAUNCH_FAILED);
        return rc;
    }
    if (yvex_cuda_capture_active(backend))
        return yvex_cuda_graph_kernel_capture(
            backend, variant, function, grid_x, block_x, shared_bytes, where, err);
    return YVEX_OK;
}

int yvex_cuda_synchronize(yvex_backend *backend,
                          yvex_backend_operation_variant variant,
                          const char *where,
                          yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    rc = backend_dispatch_admit(backend, where, err);
    if (rc != YVEX_OK) return rc;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, where, "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    if (yvex_cuda_capture_active(backend)) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (cuda_test_failure_matches("YVEX_TEST_CUDA_SYNC_FAILURE", variant)) {
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED);
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA synchronization failure");
        return YVEX_ERR_BACKEND;
    }
    rc = yvex_cuda_status(&state->driver, state->driver.cuCtxSynchronize(), where, err);
    if (rc != YVEX_OK) {
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED);
    }
    return rc;
}

/*
 * Complete work on the launch stream without stalling unrelated CUDA work.
 *
 * Older admitted CUDA contexts may not own a launch stream. They retain the exact context-wide
 * fallback and report it to the caller so physical accounting cannot disguise the wider barrier.
 */
int yvex_cuda_launch_synchronize(yvex_backend *backend,
                                 yvex_backend_operation_variant variant,
                                 int *device_wide,
                                 const char *where,
                                 yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUstream stream;
    int rc;
    if (device_wide) *device_wide = 0;
    rc = backend_dispatch_admit(backend, where, err);
    if (rc != YVEX_OK) return rc;
    if (!state || !device_wide) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where,
                       "CUDA launch synchronization owner is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    if (yvex_cuda_capture_active(backend)) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    stream = yvex_cuda_launch_stream(backend);
    if (!stream || !state->driver.cuStreamSynchronize) {
        rc = yvex_cuda_synchronize(backend, variant, where, err);
        if (rc == YVEX_OK) *device_wide = 1;
        return rc;
    }
    if (cuda_test_failure_matches("YVEX_TEST_CUDA_SYNC_FAILURE", variant)) {
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED);
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA synchronization failure");
        return YVEX_ERR_BACKEND;
    }
    rc = yvex_cuda_status(&state->driver, state->driver.cuStreamSynchronize(stream), where, err);
    if (rc != YVEX_OK)
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED);
    return rc;
}

static int cuda_raw_release(yvex_backend *backend,
                            yvex_backend_operation_variant variant,
                            CUdeviceptr pointer,
                            const char *where,
                            yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    if (cuda_test_failure_matches("YVEX_TEST_CUDA_CLEANUP_FAILURE", variant)) {
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_CLEANUP_FAILED);
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA temporary cleanup failure before release");
        return YVEX_ERR_BACKEND;
    }
    rc = yvex_cuda_status(&state->driver, state->driver.cuMemFree_v2(pointer), where, err);
    if (rc != YVEX_OK)
        cuda_capability_fail(backend, variant,
                             YVEX_BACKEND_CAPABILITY_REASON_CLEANUP_FAILED);
    return rc;
}

static int cuda_deferred_release_adopt(yvex_cuda_backend_state *state,
                                       CUdeviceptr pointer,
                                       unsigned long long bytes,
                                       yvex_backend_operation_variant variant)
{
    yvex_cuda_deferred_release *entry;
    unsigned int index;
    for (index = 0u; index < state->deferred_release_count; ++index) {
        entry = &state->deferred_releases[index];
        if (entry->pointer == pointer)
            return entry->bytes == bytes && entry->variant == variant;
    }
    if (state->deferred_release_count >= YVEX_CUDA_DEFERRED_RELEASE_MAX ||
        state->deferred_release_bytes > ULLONG_MAX - bytes)
        return 0;
    entry = &state->deferred_releases[state->deferred_release_count++];
    entry->pointer = pointer;
    entry->bytes = bytes;
    entry->variant = variant;
    state->deferred_release_bytes += bytes;
    return 1;
}
/* Contract: releases one accounted allocation or transfers it to retryable backend ownership. */

int yvex_cuda_temporary_free(yvex_backend *backend,
                             yvex_backend_operation_variant variant,
                             CUdeviceptr *ptr,
                             unsigned long long bytes,
                             int defer_on_failure,
                             const char *where,
                             yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_error release_error;
    int rc;
    if (!ptr || !*ptr) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!state || !bytes) {
        yvex_error_set(err, YVEX_ERR_STATE, where,
                       "CUDA temporary ownership metadata is missing");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(&release_error);
    rc = cuda_raw_release(backend, variant, *ptr, where, &release_error);
    if (rc == YVEX_OK) {
        backend_memory_release(backend, bytes);
        *ptr = 0u;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (defer_on_failure) {
        if (cuda_deferred_release_adopt(state, *ptr, bytes, variant)) {
            *ptr = 0u;
        } else {
            yvex_error_set(err, YVEX_ERR_NOMEM, where,
                           "CUDA deferred-release registry capacity is exhausted");
            return YVEX_ERR_NOMEM;
        }
    }
    if (err)
        *err = release_error;
    return rc;
}

int yvex_cuda_deferred_release_drain(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_error release_error;
    unsigned long long retained_bytes = 0ull;
    unsigned int index, retained = 0u;
    int result = YVEX_OK;
    int rc;
    if (!state || state->deferred_release_count == 0u) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_cuda_set_current(backend, "cuda.deferred_release.context", err);
    if (rc != YVEX_OK)
        return rc;
    for (index = 0u; index < state->deferred_release_count; ++index) {
        yvex_cuda_deferred_release entry = state->deferred_releases[index];
        yvex_error_clear(&release_error);
        rc = cuda_raw_release(backend, entry.variant, entry.pointer,
                              "cuda.deferred_release.drain", &release_error);
        if (rc == YVEX_OK) {
            backend_memory_release(backend, entry.bytes);
            continue;
        }
        state->deferred_releases[retained++] = entry;
        retained_bytes += entry.bytes;
        if (result == YVEX_OK) {
            result = rc;
            if (err)
                *err = release_error;
        }
    }
    while (index > retained) {
        --index;
        memset(&state->deferred_releases[index], 0,
               sizeof(state->deferred_releases[index]));
    }
    state->deferred_release_count = retained;
    state->deferred_release_bytes = retained_bytes;
    if (result == YVEX_OK)
        yvex_error_clear(err);
    return result;
}

static int attention_configuration_reject(yvex_error *err, yvex_status status,
                                          const char *message)
{
    yvex_error_set(err, status, "cuda.attention.configure", message);
    return status;
}

/*
 * Configuration records are immutable because captured capacity is part of executable topology.
 * Phase selection may revisit an admitted record, while an execution-identity change invalidates
 * graph executables before the new identity becomes active.
 */
int yvex_backend_cuda_attention_configure(
    yvex_backend *backend, yvex_backend_attention_phase phase,
    yvex_backend_cuda_attention_mode mode,
    const char *compatibility_identity, const char *capture_bucket,
    unsigned long long local_capacity, unsigned long long compressed_capacity,
    unsigned long long indexer_capacity, yvex_error *err)
{
    yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    const yvex_cuda_attention_configuration *active = NULL;
    yvex_cuda_attention_configuration *configuration;
    yvex_backend_cuda_graph_capability capability;
    unsigned int index;
    int rc;
    if (!state || phase < YVEX_BACKEND_ATTENTION_PHASE_DECODE ||
        phase >= YVEX_BACKEND_ATTENTION_PHASE_COUNT ||
        mode < YVEX_BACKEND_CUDA_ATTENTION_EAGER ||
        mode > YVEX_BACKEND_CUDA_ATTENTION_FULL || !compatibility_identity ||
        !compatibility_identity[0] || !capture_bucket || !capture_bucket[0] ||
        strlen(compatibility_identity) >=
            sizeof(state->attention_configurations[0].compatibility_identity) ||
        strlen(capture_bucket) >=
            sizeof(state->attention_configurations[0].capture_bucket))
        return attention_configuration_reject(
            err, YVEX_ERR_INVALID_ARG,
            "CUDA backend, phase, concrete mode, identity, and capture bucket are required");
    rc = backend_dispatch_admit(backend, "cuda.attention.configure", err);
    if (rc != YVEX_OK) return rc;
    if (mode != YVEX_BACKEND_CUDA_ATTENTION_EAGER) {
        rc = yvex_backend_cuda_graph_query(backend, &capability, err);
        if (rc != YVEX_OK) return rc;
        if (capability.state != YVEX_BACKEND_CUDA_GRAPH_OPEN ||
            !capability.edge_inventory_available || !capability.async_copy_available ||
            !capability.pinned_host_memory_available) {
            yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "cuda.attention.configure",
                            "CUDA graph topology/copy/staging admission failed: reason=%u",
                            (unsigned int)capability.reason);
            return YVEX_ERR_UNSUPPORTED;
        }
        if (!backend->workspace_device_tensor || !backend->resident_device_tensor)
            return attention_configuration_reject(
                err, YVEX_ERR_STATE,
                "CUDA graph attention requires stable workspace and resident weights");
    }
    if (state->attention_configuration_count &&
        state->attention_active_configuration < state->attention_configuration_count)
        active = &state->attention_configurations[state->attention_active_configuration];
    if (active && strcmp(active->compatibility_identity, compatibility_identity) != 0) {
        unsigned long long affected;
        rc = yvex_backend_cuda_attention_graph_registry_apply(
            backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &affected, err);
        if (rc != YVEX_OK) return rc;
    }
    for (index = 0u; index < state->attention_configuration_count; ++index) {
        configuration = &state->attention_configurations[index];
        if (configuration->phase == phase && configuration->mode == mode &&
            configuration->local_capacity == local_capacity &&
            configuration->compressed_capacity == compressed_capacity &&
            configuration->indexer_capacity == indexer_capacity &&
            strcmp(configuration->compatibility_identity, compatibility_identity) == 0 &&
            strcmp(configuration->capture_bucket, capture_bucket) == 0) {
            state->attention_active_configuration = index;
            state->attention_configuration_hits++;
            yvex_error_clear(err);
            return YVEX_OK;
        }
    }
    if (state->attention_configuration_count >= YVEX_CUDA_ATTENTION_CONFIGURATION_CAP)
        return attention_configuration_reject(
            err, YVEX_ERR_BOUNDS, "CUDA attention configuration registry is full");
    index = state->attention_configuration_count++;
    configuration = &state->attention_configurations[index];
    memset(configuration, 0, sizeof(*configuration));
    configuration->configured = 1;
    configuration->phase = phase;
    configuration->mode = mode;
    configuration->local_capacity = local_capacity;
    configuration->compressed_capacity = compressed_capacity;
    configuration->indexer_capacity = indexer_capacity;
    memcpy(configuration->compatibility_identity, compatibility_identity,
           strlen(compatibility_identity) + 1u);
    memcpy(configuration->capture_bucket, capture_bucket, strlen(capture_bucket) + 1u);
    state->attention_active_configuration = index;
    state->attention_configuration_misses++;
    yvex_error_clear(err);
    return YVEX_OK;
}
