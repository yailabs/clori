/* Selective scalar-transition structured state-space duality (SSD).
 * Families supply projections and numeric policy; callers own state and workspace. */
#ifndef INCLUDE_YVEX_INTERNAL_STATE_SPACE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_STATE_SPACE_H_INCLUDED

#include <yvex/internal/sequence_mixer.h>
#include <yvex/internal/semantic_decoder.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned long long token_count;
    /* Row order: gate[width], x[width], B[groups,state], C[groups,state], dt[heads]. */
    const float *projection;
    unsigned long long projection_capacity;
    const float *convolution_weight, *convolution_bias;
    unsigned long long convolution_weight_capacity, convolution_bias_capacity;
    const float *decay_log, *skip, *time_bias, *normalization_weight;
    unsigned long long decay_log_capacity, skip_capacity, time_bias_capacity;
    unsigned long long normalization_weight_capacity;
    yvex_sequence_state_view state;
    yvex_sequence_state_output next_state;
    float *workspace, *output;
    unsigned long long workspace_capacity, output_capacity;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_selective_ssd_cpu_request;

typedef struct {
    unsigned long long completed_tokens, state_updates;
    int complete, cancelled;
} yvex_selective_ssd_cpu_result;

/* Serial F32 reference lowering. Multi-token execution is a scan, not a chunk-parallel kernel.
 * Workspace requires convolution_width F32 values and is not persistent state.
 * Next-state and output may be partial on failure; committed state is never writable. */
int yvex_selective_ssd_execute_cpu(
    const yvex_selective_ssd_geometry *geometry, const yvex_selective_ssd_cpu_request *request,
    yvex_selective_ssd_cpu_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
