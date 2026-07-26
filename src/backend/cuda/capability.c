/* Owner: src/backend/cuda
 * Owns: canonical kernel-bundle admission, exact CUDA operation variants, atomic module/function rollback,
 *   launch/synchronization failure state, and temporary CUDA allocation cleanup.
 * Does not own: tensor geometry, graph semantics, qtype compute, model-family behavior, runtime generation, CLI
 *   rendering, or generated PTX storage.
 * Invariants: no kernel variant is supported before the canonical generated bundle and every required function are
 *   admitted; failed launch or synchronization never remains a supported variant.
 * Boundary: bounded primitive capability is not transformer or generation support.
 * Purpose: Admit the exact generated kernel bundle and mediate fail-closed CUDA launches.
 * Inputs: A context-ready CUDA backend, generated PTX bytes, and typed operation requests.
 * Effects: Loads or unloads the owned module and updates capability state atomically.
 * Failure: Rejects missing symbols, launch errors, and synchronization failures without partial capability. */

#include "src/backend/cuda/private.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef YVEX_HAVE_CUDA_KERNEL_PTX
#include <cuda_kernels_ptx.inc>
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
    {"yvex_rope_f32", YVEX_BACKEND_VARIANT_ROPE_F32, CUDA_HANDLE_OFFSET(rope_function)},
    {"yvex_matmul_f32", YVEX_BACKEND_VARIANT_MATMUL_F32,
     CUDA_HANDLE_OFFSET(matmul_function)},
    {"yvex_qtype_row_dot", YVEX_BACKEND_VARIANT_QTYPE_ROW_DOT,
     CUDA_HANDLE_OFFSET(qtype_row_dot_function)},
    {"yvex_attention_bf16_round", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(attention_bf16_round_function)},
    {"yvex_deepseek_qtype_matvec", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(deepseek_qtype_matvec_function)},
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
    {"yvex_moe_swiglu", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_swiglu_function)},
    {"yvex_moe_accumulate", YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
     CUDA_HANDLE_OFFSET(moe_accumulate_function)},
    {"yvex_mlp_f32", YVEX_BACKEND_VARIANT_MLP_DENSE_F32,
     CUDA_HANDLE_OFFSET(mlp_function)},
    {"yvex_attention_f32", YVEX_BACKEND_VARIANT_ATTENTION_CAUSAL_F32,
     CUDA_HANDLE_OFFSET(attention_function)},
};
#define CUDA_KERNEL_BINDING_COUNT (sizeof(cuda_kernel_bindings) / sizeof(cuda_kernel_bindings[0]))

#undef CUDA_HANDLE_OFFSET

/* Purpose: address one table-admitted function slot without exposing state layout. */
static CUfunction *cuda_function_slot(yvex_cuda_backend_state *state, size_t offset)
{
    return (CUfunction *)((unsigned char *)state + offset);
}

/* Purpose: clear every module/function handle without Driver API side effects. */
static void cuda_bundle_clear_handles(yvex_cuda_backend_state *state)
{
    size_t index;

    if (!state) {
        return;
    }
    state->module = NULL;
    for (index = 0; index < CUDA_KERNEL_BINDING_COUNT; ++index) {
        *cuda_function_slot(state, cuda_kernel_bindings[index].state_offset) = NULL;
    }
    state->module_loaded = 0;
    state->kernel_bundle_identity[0] = '\0';
}

#ifdef YVEX_HAVE_CUDA_KERNEL_PTX
/* Purpose: identify the exact generated PTX bytes admitted by this process. */
static int cuda_bundle_identity(char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.cuda.kernel-bundle.v1") ||
        !yvex_sha256_update(&hash, cuda_kernels_ptx,
                            strlen((const char *)cuda_kernels_ptx)) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
#endif

/* Contract: reports whether an explicit test-only failure selector matches. */
/* Purpose: Implement the canonical test failure matches mechanism owned by the CUDA backend boundary. */
static int cuda_test_failure_matches(const char *name,
                                     yvex_backend_operation_variant variant)
{
    const char *value = getenv(name);

    return value && value[0] != '\0' &&
           (strcmp(value, "all") == 0 ||
            strcmp(value, yvex_backend_operation_variant_name(variant)) == 0);
}

/* Contract: returns the admitted function handle for one exact kernel variant. */
/* Purpose: Implement the canonical variant function mechanism owned by the CUDA backend boundary.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Purpose: publish every initialized field of one deterministic capability projection.
 * Inputs: caller-owned result, typed state and reason, and exact function availability.
 * Effects: initializes the varying result fields and clears the caller error.
 * Failure: this total projection cannot fail after its caller validates the result owner.
 * Boundary: does not inspect CUDA state or infer capability policy. */
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

/* Contract: records one execution failure without changing unrelated variants. */
/* Purpose: Implement the canonical capability fail mechanism owned by the CUDA backend boundary. */
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
/* Purpose: retain a rejected module under an already valid context owner.
 * Inputs: context-owning backend and the unpublished module returned by the Driver.
 * Effects: clears every function handle and records only the module for checked close.
 * Failure: this ownership transfer cannot fail after backend admission.
 * Boundary: rejected functions never become executable; checked close discharges the module. */
static void cuda_bundle_retain_rejected(yvex_backend *backend, CUmodule module)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);

    cuda_bundle_clear_handles(state);
    if (state && module) {
        state->module = module;
        state->module_loaded = 1;
    }
}

/* Contract: resolves one required function or returns a typed atomic-admission failure. */
/* Purpose: Implement the canonical resolve required mechanism owned by the CUDA backend boundary.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
static int cuda_resolve_required(yvex_cuda_backend_state *state,
                                 CUmodule module,
                                 const char *symbol,
                                 yvex_backend_operation_variant variant,
                                 CUfunction *out,
                                 yvex_error *err)
{
    const char *injected = getenv("YVEX_TEST_CUDA_BUNDLE_FAILURE");

    *out = NULL;
    if (injected &&
        (strcmp(injected, "symbol") == 0 || strcmp(injected, symbol) == 0 ||
         strcmp(injected, yvex_backend_operation_variant_name(variant)) == 0)) {
        yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.kernels.resolve",
                        "required CUDA function unavailable: %s", symbol);
        return YVEX_ERR_BACKEND;
    }
    return yvex_cuda_status(&state->driver,
                            state->driver.cuModuleGetFunction(out, module, symbol),
                            "cuda.kernels.resolve", err);
}
#endif

/*
 * Contract: admits only generated kernels.cu PTX and commits no handle
 * until module load and every required symbol succeed. No tensor payload IO. */
/* Purpose: Implement the canonical kernel bundle admit mechanism owned by the CUDA backend boundary.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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
    if (state->module || state->module_loaded) {
        if (state->module && state->module_loaded &&
            state->kernel_bundle_state == YVEX_CUDA_KERNEL_BUNDLE_ADMITTED) {
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
        CUmodule module = NULL;
        CUfunction functions[CUDA_KERNEL_BINDING_COUNT];
        const char *injected = getenv("YVEX_TEST_CUDA_BUNDLE_FAILURE");
        size_t index;
        int rc;

        memset(functions, 0, sizeof(functions));
        rc = yvex_cuda_set_current(backend, "cuda.kernels.admit", err);
        if (rc != YVEX_OK) {
            state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
            state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_CONTEXT_UNAVAILABLE;
            return rc;
        }
        if (injected && strcmp(injected, "module") == 0) {
            yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.kernels.load",
                           "injected CUDA module admission failure");
            rc = YVEX_ERR_BACKEND;
        } else {
            rc = yvex_cuda_status(&state->driver,
                                  state->driver.cuModuleLoadData(
                                      &module, cuda_kernels_ptx),
                                  "cuda.kernels.load", err);
        }
        if (rc != YVEX_OK) {
            state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
            state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED;
            goto reject;
        }

        for (index = 0; index < CUDA_KERNEL_BINDING_COUNT; ++index) {
            const cuda_kernel_binding *binding = &cuda_kernel_bindings[index];

            rc = cuda_resolve_required(state, module, binding->symbol, binding->variant,
                                       &functions[index], err);
            if (rc != YVEX_OK) {
                state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_REJECTED;
                state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING;
                state->kernel_bundle_failure_variant = binding->variant;
                goto reject;
            }
        }

        state->module = module;
        for (index = 0; index < CUDA_KERNEL_BINDING_COUNT; ++index) {
            *cuda_function_slot(state, cuda_kernel_bindings[index].state_offset) = functions[index];
        }
        state->module_loaded = 1;
        state->kernel_bundle_state = YVEX_CUDA_KERNEL_BUNDLE_ADMITTED;
        state->kernel_bundle_reason = YVEX_BACKEND_CAPABILITY_REASON_NONE;
        state->kernel_bundle_failure_variant = YVEX_BACKEND_VARIANT_COUNT;
        if (!cuda_bundle_identity(state->kernel_bundle_identity)) {
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
        cuda_bundle_retain_rejected(backend, module);
        return rc;
    }
#endif
}

/* Contract: unloads the admitted module and preserves every handle on pre-release failure. */
/* Purpose: Release the resources owned by kernel bundle close without changing borrowed inputs.
 * Inputs: An owned object that may be null or already released where its lifecycle permits.
 * Effects: Releases only resources owned by the supplied object and leaves it reset or unusable.
 * Failure: Null and already-released inputs follow the idempotent lifecycle contract.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
int yvex_cuda_kernel_bundle_close(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc = YVEX_OK;

    if (!backend || !state) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (state->module_loaded && state->module && state->driver.cuModuleUnload) {
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuModuleUnload(state->module),
                              "cuda.kernels.unload", err);
    } else if (state->module_loaded && state->module) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "cuda.kernels.unload",
                       "CUDA module unload function is unavailable");
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

/* Contract: projects one exact variant without loading modules or executing work. */
/* Purpose: Project the admitted numeric and device capability for query capability.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Does not mutate caller-visible or owner state.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Contract: refuses an unsupported or failed variant before any CUDA dispatch. */
/* Purpose: Project the admitted numeric and device capability for require capability.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Contract: launches one admitted function and records launch failure atomically. */
/* Purpose: Execute the typed launch operation over already admitted buffers.
 * Inputs: Typed admitted handles, immutable source ranges, checked dimensions, and an explicit destination.
 * Effects: Mutates only the admitted destination or transaction after every precondition passes.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Contract: synchronizes one variant and demotes it on any synchronization failure. */
/* Purpose: Implement the canonical synchronize mechanism owned by the CUDA backend boundary.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Purpose: perform one true Driver release without changing ownership metadata.
 * Inputs: live backend, admitted variant, nonzero device pointer, and failure context.
 * Effects: calls the Driver exactly once unless the explicit pre-release seam refuses.
 * Failure: demotes the exact capability and leaves the allocation physically owned.
 * Boundary: accounting and ownership transfer remain with the temporary-release owner. */
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

/* Purpose: transfer a failed stack-local release into the backend's retryable owner.
 * Inputs: live CUDA state and exact pointer, byte extent, and operation variant.
 * Effects: records one unique deferred allocation without changing live accounting.
 * Failure: false on conflicting duplicate, capacity exhaustion, or byte overflow.
 * Boundary: adoption records ownership only; it never calls the Driver or clears a caller pointer. */
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
/* Purpose: discharge one raw CUDA allocation without losing a pointer after Driver refusal.
 * Inputs: live backend, exact allocation identity/bytes, and explicit deferred-ownership policy.
 * Effects: clears the caller pointer only after true release or successful backend adoption.
 * Failure: pre-release failure retains accounting and either caller or deferred ownership.
 * Boundary: CUDA allocation lifecycle; it does not release workspace-backed ranges. */
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

/* Purpose: retry every backend-owned deferred allocation in stable acquisition order.
 * Inputs: a live CUDA backend retaining zero or more failed raw releases.
 * Effects: removes and unaccounts only allocations physically released by the Driver.
 * Failure: retains every failed entry for a later checked-close retry.
 * Boundary: checked backend teardown and pre-work admission are the only consumers. */
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
